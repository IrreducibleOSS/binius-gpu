# AGENTS.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Experimental CUDA GPU kernels for select computations needed by a [Binius](https://github.com/IrreducibleOSS/binius) prover: binary tower field multiplication (compact and bit-sliced), the additive NTT, and sumcheck over the 128-bit binary tower field. Not used by the official Binius prover.

This is a CUDA C++ project (C++20, CUDA standard 20), built with CMake. Header logic lives in `.cuh` files; most kernels and math are `__host__ __device__` templates so they can be unit-tested on CPU and run on GPU.

## Build

```
cmake -B./build -DCMAKE_BUILD_TYPE=Release
cmake --build ./build
```

Submodules must be present first: `git submodule update --init --recursive` (Catch2 and nvbench under `third-party/`).

`Release` defines `STRIP_ASSERTIONS` and `STRIP_CUDA_CHECK`, which compile out the `ASSERT(...)` and `CUDA_CHECK(...)` macros (see `src/ulvt/utils/common.cuh`). Use `-DCMAKE_BUILD_TYPE=Debug` to keep those checks active when diagnosing failures.

## Test and benchmark

Tests use **Catch2**; standalone benchmarks use **nvbench**. Executables land in `./build/`.

```
./build/finite_field_tests          # Catch2: field arithmetic
./build/ntt_tests                   # Catch2: additive NTT (tests + benches together)
./build/sumcheck_test               # Catch2: binary-tower sumcheck
./build/sumcheck_bench              # binary-tower sumcheck benchmark
./build/prime_fields_gpu            # Catch2: prime-field (QM31) sumcheck
./build/gpu-benchmarks              # nvbench: field-multiply throughput
./build/circuit_generator           # emits unrolled tower-multiply circuits
```

Run a single Catch2 test case or section by name/tag, e.g. `./build/finite_field_tests "<test name or tag>"`; `--list-tests` enumerates them. nvbench binaries accept `--help` for filtering/output options.

## Architecture

CMake builds four static libraries from `src/ulvt/`:

- **`ulvt_gpu`** — finite fields (`finite_fields/`), additive NTT (`ntt/`), shared utils (`utils/`). Public include dir is `./src`, so headers are included as `ulvt/...`. Link against this for NTT and field ops.
- **`sumcheck`** — binary-tower sumcheck (`sumcheck/`). Link against this for sumcheck. Depends on `unrolled`.
- **`prime_field_sumcheck`** — a separate sumcheck over the QM31 prime-field tower (`prime_field_sumcheck/`). Independent of the binary-tower path.
- **`unrolled`** — machine-generated, fully unrolled tower-field multiply circuits (`finite_fields/circuit_generator/unrolled/`), produced by `circuit_generator`. Linked privately into `ulvt_gpu` and `sumcheck`.

All CUDA targets use `CUDA_SEPARABLE_COMPILATION ON` and compile with `--use_fast_math --generate-line-info --relocatable-device-code=true`.

### Finite fields (`src/ulvt/finite_fields/`)

- `binary_tower.cuh` — `FanPaarTowerField<HEIGHT>`, a recursive Fan-Paar tower of GF(2^(2^HEIGHT)) (HEIGHT ≤ 5, i.e. up to 32-bit). `multiply` recurses via Karatsuba (`generic_multiply`) down to the base field. Addition is XOR.
- `binary_tower_simd.cuh` — bit-sliced field arithmetic (operate on 32 elements in parallel across the bits of `uint32_t` lanes).
- `ghash_ctmul32.cuh` / `ghash_ctmul64.cuh` — GF(2^128) as F[X]/(X^128 + X^7 + X^2 + X + 1), carryless-multiply implementations ported from BearSSL.
- `circuit_generator/` — a CPU tool that generates the unrolled multiply circuits in `unrolled/`.
- `baby_bear.cuh`, `m31.cuh`, `cm31.cuh`, `qm31.cuh` — prime fields and extensions used by the prime-field sumcheck.

### Bit-slicing (`src/ulvt/utils/bitslicing.cuh`)

Central to the GPU layout. `BitsliceUtils<BITS_WIDTH>` transposes between *traditional* storage (consecutive ints per value) and *bit-sliced* storage (a block of `BITS_WIDTH` ints holds one bit-position across many values). All bit-slicing is little-endian. The 128-bit tower (`TOWER_HEIGHT 7`) gives `BITS_WIDTH = 128`, `INTS_PER_VALUE = 4`; a bit-sliced block of 128 `uint32_t`s represents 32 field elements.

### Sumcheck (`src/ulvt/sumcheck/`)

`Sumcheck<NUM_VARS, COMPOSITION_SIZE, DATA_IS_TRANSPOSED>` in `sumcheck.cuh` (NUM_VARS ∈ {20,24,28}, COMPOSITION_SIZE ∈ {2,3,4}). The `DATA_IS_TRANSPOSED` template flag selects the input layout:
- `true` → `evals_span` holds bit-sliced blocks of 128 32-bit ints (32 elements of F(2^128) each).
- `false` → `evals_span` holds blocks of 4 consecutive 32-bit ints (one F(2^128) element each, traditional layout).

The round logic folds multilinear evaluation lists in half each round (`fold_list_halves` dispatches to a GPU kernel for large lists, CPU `fold_small` for small ones); kernels live in `sumcheck/core/`. Shared compile-time constants (tower height, widths, grid dims `BLOCKS`/`THREADS_PER_BLOCK`) are in `sumcheck/utils/constants.hpp`.

### Additive NTT (`src/ulvt/ntt/`)

`additive_ntt.cuh` implements the in-place additive-NTT butterfly over binary tower fields, parameterized by field type. Configuration in `nttconf.{cuh,cu}`.

## Conventions

- `.cu`/`.cuh` files use 4-space indentation (`.editorconfig`).
- Prefer `__host__ __device__` for math helpers so they stay CPU-testable.
- Use the `ASSERT` / `CUDA_CHECK` macros from `utils/common.cuh` rather than raw checks (they vanish in Release builds).
