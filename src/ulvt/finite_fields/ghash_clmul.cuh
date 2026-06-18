#pragma once

#include <cstdint>
#include "ghash_ctmul64.cuh"

namespace ghash_clmul {

using ghash_ctmul64::GhashWide;

/** Elements of GF(2^128) represented as F[X] / (X^128 + X^7 + X^2 + X + 1)
 *
 * Limbs are little-endian: limbs[0] holds coefficients of X^0..X^63, limbs[1]
 * holds X^64..X^127.
 */
class Ghash {
public:
    uint64_t limbs[2];

    __host__ __device__ Ghash() : limbs{0, 0} {}
    __host__ __device__ Ghash(const uint64_t (&limbs)[2]) : limbs{limbs[0], limbs[1]} {}
    __host__ __device__ Ghash(__uint128_t value)
        : limbs{(uint64_t)value, (uint64_t)(value >> 64)} {}

    __host__ __device__ bool operator==(const Ghash& o) const {
        return limbs[0] == o.limbs[0] && limbs[1] == o.limbs[1];
    }
    __host__ __device__ bool operator!=(const Ghash& o) const { return !(*this == o); }

    // Addition and subtraction in GF(2^128) are both XOR.
    __host__ __device__ Ghash& operator+=(const Ghash& o) {
        limbs[0] ^= o.limbs[0];
        limbs[1] ^= o.limbs[1];
        return *this;
    }
    __host__ __device__ Ghash& operator-=(const Ghash& o) { return *this += o; }
    __host__ __device__ Ghash operator+(const Ghash& o) const {
        Ghash r = *this;
        return r += o;
    }
    __host__ __device__ Ghash operator-(const Ghash& o) const { return *this + o; }

    // Defined out-of-line below, after clmad() and reduce().
    __host__ __device__ Ghash operator*(const Ghash& o) const;
    __host__ __device__ Ghash& operator*=(const Ghash& o);
};

__host__ __device__
static inline uint64_t clmad64_lo(uint64_t a, uint64_t b, uint64_t c) {
  // PTX ISA 9.3 ships with CUDA Toolkit 13.3. Toolkit version increases
  // monotonically with PTX ISA version, so ">= 13.3" == "PTX ISA >= 9.3".
  // clmad instruction requires sm_80 or higher
  // See: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-clmad
#if defined(__CUDA_ARCH__) &&                                               \
    __CUDA_ARCH__ >= 800 &&                                                 \
    defined(__CUDACC_VER_MAJOR__) &&                                        \
    (__CUDACC_VER_MAJOR__ > 13 ||                                           \
      (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 3))
    uint64_t d;
    asm("clmad.lo.u64 %0, %1, %2, %3;" : "=l"(d) : "l"(a), "l"(b), "l"(c));
    return d;
#else
    return ghash_ctmul64::bmul64(a, b) ^ c;
#endif
}

__host__ __device__
static inline uint64_t clmad64_hi(uint64_t a, uint64_t b, uint64_t c) {
  // PTX ISA 9.3 ships with CUDA Toolkit 13.3. Toolkit version increases
  // monotonically with PTX ISA version, so ">= 13.3" == "PTX ISA >= 9.3".
  // clmad instruction requires sm_80 or higher
  // See: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-clmad
#if defined(__CUDA_ARCH__) &&                                               \
    __CUDA_ARCH__ >= 800 &&                                                 \
    defined(__CUDACC_VER_MAJOR__) &&                                        \
    (__CUDACC_VER_MAJOR__ > 13 ||                                           \
      (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 3))
    uint64_t d;
    asm("clmad.hi.u64 %0, %1, %2, %3;" : "=l"(d) : "l"(a), "l"(b), "l"(c));
    return d;
#else
    using ghash_ctmul64::rev64;
    using ghash_ctmul64::bmul64;

    return (rev64(bmul64(rev64(a), rev64(b))) >> 1) ^ c;
#endif
}

__host__ __device__
static void clmad(const Ghash& lhs, const Ghash& rhs, GhashWide& out) {
	const auto x0 = lhs.limbs[0];
	const auto x1 = lhs.limbs[1];
    const auto y0 = rhs.limbs[0];
    const auto y1 = rhs.limbs[1];

    const auto x2 = x0 ^ x1;
    const auto y2 = y0 ^ y1;

    const auto z0 = clmad64_lo(x0, y0, 0);
    const auto z1 = clmad64_lo(x1, y1, 0);
    auto z2 = clmad64_lo(x2, y2, 0);

	auto z0h = clmad64_hi(x0, y0, 0);
	auto z1h = clmad64_hi(x1, y1, 0);
	auto z2h = clmad64_hi(x2, y2, 0);

	z2 ^= z0 ^ z1;
	z2h ^= z0h ^ z1h;

	const auto v0 = z0;
	const auto v1 = z0h ^ z2;
	const auto v2 = z1 ^ z2h;
	const auto v3 = z1h;

    out.limbs[0] ^= v0;
    out.limbs[1] ^= v1;
    out.limbs[2] ^= v2;
    out.limbs[3] ^= v3;
}

/** Reduce modulo X^128 + X^7 + X^2 + X + 1. */
__host__ __device__
static inline Ghash reduce(GhashWide x) {
    const auto ret = ghash_ctmul64::reduce(x);
    return Ghash(ret.limbs);
}

/** Multiply in GF(2^128): carryless multiply-accumulate into a wide product,
 * then reduce modulo X^128 + X^7 + X^2 + X + 1. */
__host__ __device__ inline Ghash Ghash::operator*(const Ghash& o) const {
    GhashWide wide;  // zero-initialized; clmad accumulates with ^=
    clmad(*this, o, wide);
    // Qualified: GhashWide is ghash_ctmul64::GhashWide, so an unqualified call
    // is ambiguous between this reduce() and ghash_ctmul64::reduce() via ADL.
    return ghash_clmul::reduce(wide);
}

__host__ __device__ inline Ghash& Ghash::operator*=(const Ghash& o) {
    *this = *this * o;
    return *this;
}

}; // namespace ghash_clmul

