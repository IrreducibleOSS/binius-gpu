// Finite-field multiplication throughput benchmarks (GPU) built on nvbench.
//
// Field multiplies are tiny, fully-inlined operations, so timing a single one
// is meaningless. Instead we time a *sequence* of multiplies. The strategy
// (ported from a Rust harness that gave good results) keeps a small register
// batch of elements and repeatedly multiplies strided pairs in place. The
// stride is half the batch, which breaks the serial dependency chain into
// `Batch/2` independent chains so the hardware can pipeline them -- we measure
// throughput, not single-multiply latency. `Passes` controls how many multiply
// rounds happen per timed invocation.

#include <array>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>

#include <nvbench/nvbench.cuh>

#include "ulvt/finite_fields/ghash_clmul.cuh"
#include "ulvt/finite_fields/ghash_ctmul32.cuh"
#include "ulvt/finite_fields/ghash_ctmul64.cuh"
#include "ulvt/finite_fields/m31.cuh"

namespace {

// ----------------------------------------------------------------------------
// The core unit of work.
//
// Multiplies strided pairs of `batch` in place for `n_passes` rounds. The inner
// loop is unrolled so the `(i + Batch/2) % Batch` indices are compile-time
// constants; that is what keeps `batch` in registers (a register array indexed
// by a runtime value would spill to local memory). Keep this pure -- the store
// that defeats dead-code elimination lives in the kernel.
template <typename T, int Batch>
__device__ inline void mul_passes(T* batch, int n_passes)
{
    for (int p = 0; p < n_passes; ++p) {
#pragma unroll
        for (int i = 0; i < Batch; ++i) {
            batch[i] = batch[i] * batch[(i + Batch / 2) % Batch];
        }
    }
}

// Fold a batch down to a single field element using field addition. Produces a
// cheap, escape-resistant summary of the final batch so the optimizer cannot
// prove the multiplies are dead. (Named `fold`, not `reduce`, to avoid clashing
// with the GF(2^128) `reduce` the ghash headers define.)
template <typename T, int Batch>
__device__ inline T fold(const T* batch)
{
    T acc = batch[0];
#pragma unroll
    for (int i = 1; i < Batch; ++i) {
        acc = acc + batch[i];
    }
    return acc;
}

// ----------------------------------------------------------------------------
// Random element generation (host side).

uint32_t random_limb(std::mt19937& rng)
{
    // Uniform over [0, P) = [0, 2^31 - 1).
    std::uniform_int_distribution<uint32_t> dist(0, M31::P - 1);
    return dist(rng);
}

template <typename T>
T random_elem(std::mt19937& rng);

template <>
M31 random_elem<M31>(std::mt19937& rng)
{
    return M31(random_limb(rng));
}

template <>
QM31 random_elem<QM31>(std::mt19937& rng)
{
    M31 limbs[4] = {M31(random_limb(rng)), M31(random_limb(rng)),
                    M31(random_limb(rng)), M31(random_limb(rng))};
    return QM31(CM31(limbs[0], limbs[1]), CM31(limbs[2], limbs[3]));
}

// Every 128-bit pattern is a valid GF(2^128) element, so draw full-width words.
template <>
ghash_ctmul32::Ghash random_elem<ghash_ctmul32::Ghash>(std::mt19937& rng)
{
    uint32_t limbs[4] = {static_cast<uint32_t>(rng()), static_cast<uint32_t>(rng()),
                         static_cast<uint32_t>(rng()), static_cast<uint32_t>(rng())};
    return ghash_ctmul32::Ghash(limbs);
}

template <>
ghash_ctmul64::Ghash random_elem<ghash_ctmul64::Ghash>(std::mt19937& rng)
{
    auto word = [&] { return (static_cast<uint64_t>(rng()) << 32) | rng(); };
    uint64_t limbs[2] = {word(), word()};
    return ghash_ctmul64::Ghash(limbs);
}

template <>
ghash_clmul::Ghash random_elem<ghash_clmul::Ghash>(std::mt19937& rng)
{
    auto word = [&] { return (static_cast<uint64_t>(rng()) << 32) | rng(); };
    uint64_t limbs[2] = {word(), word()};
    return ghash_clmul::Ghash(limbs);
}

// ----------------------------------------------------------------------------
// GPU benchmark.

// Abort the benchmark if a CUDA call failed. Deliberately not the repo's
// CUDA_CHECK macro: that compiles to a no-op under STRIP_CUDA_CHECK (set in
// Release, which is exactly how benchmarks are built). A silent launch failure
// -- e.g. a binary built for the wrong GPU architecture -- would otherwise be
// timed as a near-instant no-op and reported as absurd throughput.
void check_cuda(cudaError_t err, const char* what)
{
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }
}

template <typename T, int Batch>
__global__ void gpu_mul_kernel(const T* __restrict__ seeds,
                               T* __restrict__ results, int n_passes)
{
    const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    T batch[Batch];
#pragma unroll
    for (int i = 0; i < Batch; ++i) {
        batch[i] = seeds[i];
    }

    mul_passes<T, Batch>(batch, n_passes);

    // Store an escape-resistant summary to global memory.
    results[tid] = fold<T, Batch>(batch);
}

template <typename T, int Batch>
void gpu_mul(nvbench::state& state)
{
    const auto n_passes = static_cast<int>(state.get_int64("Passes"));

    constexpr unsigned threads_per_block = 256;
    constexpr unsigned blocks = 1024;
    constexpr unsigned threads = threads_per_block * blocks;

    // Seed batch: identical across threads (values are irrelevant to timing).
    std::mt19937 rng(0xC0FFEEu);
    std::array<T, Batch> host_seeds;
    for (auto& e : host_seeds) {
        e = random_elem<T>(rng);
    }

    T* seeds = nullptr;
    T* results = nullptr;
    check_cuda(cudaMalloc(&seeds, sizeof(host_seeds)), "cudaMalloc(seeds)");
    check_cuda(cudaMalloc(&results, threads * sizeof(T)), "cudaMalloc(results)");
    check_cuda(cudaMemcpy(seeds, host_seeds.data(), sizeof(host_seeds), cudaMemcpyHostToDevice),
               "cudaMemcpy(seeds)");

    // Warm-up launch outside the timed region: verify the kernel actually runs
    // on this device. cudaGetLastError() catches launch-time failures (bad
    // config, no kernel image for this architecture); cudaDeviceSynchronize()
    // catches errors raised during execution.
    gpu_mul_kernel<T, Batch><<<blocks, threads_per_block>>>(seeds, results, n_passes);
    check_cuda(cudaGetLastError(), "gpu_mul_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "gpu_mul_kernel execution");

    // Each of `threads` threads performs Batch * Passes multiplies per launch.
    state.add_element_count(
        static_cast<std::size_t>(threads) * Batch * n_passes, "Muls");

    state.exec([=](nvbench::launch& launch) {
        gpu_mul_kernel<T, Batch>
            <<<blocks, threads_per_block, 0, launch.get_stream()>>>(
                seeds, results, n_passes);
    });

    cudaFree(seeds);
    cudaFree(results);
}

// ----------------------------------------------------------------------------
// Registrations. Concrete wrappers avoid the comma-in-macro problem with
// templated benchmark functions.
//
// Batch sizes are picked to give every type a comparable live-data register
// budget (~32 registers). M31 is 4 B (1 register/element) so it gets 32; QM31
// and both Ghash variants are 16 B (4 registers/element) so they get 8. Larger
// batches also mean a wider stride and so more independent multiply chains for
// ILP.
void gpu_mul_m31(nvbench::state& state) { gpu_mul<M31, 32>(state); }
void gpu_mul_qm31(nvbench::state& state) { gpu_mul<QM31, 8>(state); }
void gpu_mul_ghash_ctmul32(nvbench::state& state) { gpu_mul<ghash_ctmul32::Ghash, 8>(state); }
void gpu_mul_ghash_ctmul64(nvbench::state& state) { gpu_mul<ghash_ctmul64::Ghash, 8>(state); }
void gpu_mul_ghash_clmul(nvbench::state& state) { gpu_mul<ghash_clmul::Ghash, 8>(state); }

} // namespace

NVBENCH_BENCH(gpu_mul_m31).add_int64_axis("Passes", {512});
NVBENCH_BENCH(gpu_mul_qm31).add_int64_axis("Passes", {512});
NVBENCH_BENCH(gpu_mul_ghash_ctmul32).add_int64_axis("Passes", {512});
NVBENCH_BENCH(gpu_mul_ghash_ctmul64).add_int64_axis("Passes", {512});
NVBENCH_BENCH(gpu_mul_ghash_clmul).add_int64_axis("Passes", {512});

NVBENCH_MAIN;
