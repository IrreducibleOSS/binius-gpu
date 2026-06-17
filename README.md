# Binius GPU Kernels

This project provides experimental GPU kernels for select computations required by a [Binius](/IrreducibleOSS/binius) prover.

We currently implement:

- Binary tower field multiplication, compact and bit-sliced
- The additive NTT
- Sumcheck over the 128-bit tower field

This code is experimental and not currently used by the official Binius prover.

## Build and run with CMake

```
    cmake -B./build -DCMAKE_BUILD_TYPE=Release
    cmake --build ./build
```

### GPU target architecture

The project does not pin a CUDA architecture, so CMake fills in its default:
the oldest architecture supported by the installed CUDA toolkit (for example,
`sm_75` / Turing with CUDA 13). The resulting binary embeds PTX, so it still
runs on newer GPUs via JIT, but it is not compiled for them and the SASS is not
tuned to the device actually running it.

When building on the machine you will run on, configure for its exact
architecture so the kernels are compiled natively (this matters in particular
for the throughput benchmarks):

```
    cmake -B./build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native
```

`native` probes the local GPU; if you are cross-compiling, pass the explicit
compute capability instead (e.g. `-DCMAKE_CUDA_ARCHITECTURES=90` for Hopper).

## Run module test files and benchmarks
NTTs and Finite Field Operations (Benchmarks and tests together)
```
    ./build/ntt_tests
    ./build/finite_field_tests
```

Sumcheck (Benchmarks and tests separate)
```
    ./build/sumcheck_test
    ./build/sumcheck_bench
```

### Include our implementations in your project
NTT and Finite Field dependencies can be included by linking against the ```ulvt_gpu``` library.

Sumcheck dependencies can be included by linking against the ```sumcheck``` library.

### The Sumcheck Interface
The sumcheck interface supports both bitsliced and traditionally stored values as input.

Meaning that if you set DATA_IS_TRANSPOSED to true in your instantation of a ```Sumcheck```, then ```evals_span``` should contain bit-sliced blocks of 128 32-bit integers (with each block representing 32 elements of $F_{2^{128}}$).

If you set DATA_IS_TRANSPOSED to false in your instantation of a ```Sumcheck```, then ```evals_span``` should contain blocks of 4 consecutive 32-bit integers, with each block representing an element of $F_{2^{128}}$ (the traditional way of storing 128-bit integers).

All bit-slicing is little-endian.

## License

MIT License

Copyright (c) 2024 Irreducible Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
