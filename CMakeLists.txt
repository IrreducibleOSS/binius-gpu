cmake_minimum_required(VERSION 3.27)
project(ulvt_gpu VERSION 0.1 LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Third-party dependencies
add_subdirectory(third-party/Catch2)
add_subdirectory(third-party/nvbench)

# Relocatable device code for all of our own targets (defined below).
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# The CUDA toolchain supports C++23, but CMake exposes no cuda_std_23 compile
# feature for this nvcc, so the CUDA standard is set the property way. The C++
# standard is requested per-target via target_compile_features(cxx_std_23).
set(CMAKE_CUDA_STANDARD 23)

# ---------------------------------------------------------------------------
# Libraries
# ---------------------------------------------------------------------------

# Machine-generated, fully unrolled tower-field multiply circuits.
add_library(unrolled STATIC
    src/ulvt/finite_fields/circuit_generator/unrolled/binary_tower_unrolled2.cu
    src/ulvt/finite_fields/circuit_generator/unrolled/binary_tower_unrolled5.cu
    src/ulvt/finite_fields/circuit_generator/unrolled/binary_tower_unrolled6.cu
    src/ulvt/finite_fields/circuit_generator/unrolled/binary_tower_unrolled7.cu
    src/ulvt/finite_fields/circuit_generator/unrolled/binary_tower_unrolled.cuh
)
target_compile_features(unrolled PUBLIC cxx_std_23)

# Finite fields, additive NTT, and shared utilities.
add_library(ulvt_gpu STATIC
    src/ulvt/ntt/nttconf.cu
    src/ulvt/ntt/gpuntt.cuh
    src/ulvt/ntt/nttconf.cuh
    src/ulvt/ntt/additive_ntt.cuh
    src/ulvt/utils/common.cu
    src/ulvt/utils/common.cuh
    src/ulvt/finite_fields/baby_bear.cuh
    src/ulvt/finite_fields/risc0_baby_bear.h
    src/ulvt/finite_fields/binary_tower.cuh
    src/ulvt/finite_fields/binary_tower_simd.cuh
    src/ulvt/finite_fields/tests/profiling/kernels/babybear_repeat.cu
    src/ulvt/finite_fields/tests/profiling/kernels/babybear_repeat.cuh
    src/ulvt/finite_fields/tests/profiling/kernels/bitsliced_repeat.cu
    src/ulvt/finite_fields/tests/profiling/kernels/bitsliced_repeat.cuh
    src/ulvt/finite_fields/circuit_generator/utils/utils.cpp
    src/ulvt/finite_fields/circuit_generator/utils/utils.hpp
)
target_compile_features(ulvt_gpu PUBLIC cxx_std_23)
target_include_directories(ulvt_gpu PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(ulvt_gpu PRIVATE unrolled)

# Binary-tower sumcheck.
add_library(sumcheck STATIC
    src/ulvt/sumcheck/utils/constants.hpp
    src/ulvt/sumcheck/sumcheck.cuh
    src/ulvt/sumcheck/core/core.cu
    src/ulvt/sumcheck/core/core.cuh
    src/ulvt/sumcheck/core/kernels.cu
    src/ulvt/sumcheck/core/kernels.cuh
    src/ulvt/utils/bitslicing.cuh
)
target_compile_features(sumcheck PUBLIC cxx_std_23)
target_include_directories(sumcheck PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(sumcheck PRIVATE unrolled)

# Prime-field (QM31) sumcheck.
add_library(prime_field_sumcheck STATIC
    src/ulvt/prime_field_sumcheck/sumcheck.cuh
    src/ulvt/prime_field_sumcheck/core/kernels.cu
    src/ulvt/prime_field_sumcheck/core/kernels.cuh
    src/ulvt/prime_field_sumcheck/utils/interpolate.hpp
)
target_compile_features(prime_field_sumcheck PUBLIC cxx_std_23)

# ---------------------------------------------------------------------------
# Tests and benchmarks
# ---------------------------------------------------------------------------

add_executable(ntt_tests
    src/ulvt/ntt/tests/md5.cu
    src/ulvt/ntt/tests/test_ntt.cu
)
target_link_libraries(ntt_tests PRIVATE ulvt_gpu Catch2::Catch2WithMain)
target_compile_options(ntt_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       --use_fast_math
                       --relocatable-device-code=true
                       >)

add_executable(finite_field_tests
    src/ulvt/finite_fields/tests/test_fanpaartower.cu
    src/ulvt/finite_fields/tests/test_ghash.cu
    src/ulvt/finite_fields/tests/test_kernels.cu
    src/ulvt/finite_fields/tests/tests.cu
)
target_link_libraries(finite_field_tests PRIVATE ulvt_gpu Catch2::Catch2WithMain)
target_compile_options(finite_field_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       --use_fast_math
                       --relocatable-device-code=true
                       >)

add_executable(sumcheck_test
    src/ulvt/sumcheck/test/test.cu
    src/ulvt/sumcheck/test/kernel/verifier_kernel.cu
    src/ulvt/sumcheck/test/kernel/verifier_kernel.cuh
    src/ulvt/sumcheck/test/utils/bigints.cu
    src/ulvt/sumcheck/test/utils/bigints.cuh
    src/ulvt/sumcheck/test/utils/tower_7_mul.cu
    src/ulvt/sumcheck/test/utils/tower_7_mul.cuh
    src/ulvt/sumcheck/test/utils/unbitsliced_mul.cuh
    src/ulvt/sumcheck/test/verifier.cu
    src/ulvt/sumcheck/test/verifier.cuh
)
target_link_libraries(sumcheck_test PRIVATE sumcheck Catch2::Catch2WithMain)
target_compile_options(sumcheck_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       --use_fast_math
                       --relocatable-device-code=true
                       >)

add_executable(sumcheck_bench src/ulvt/sumcheck/bench/benchmark.cu)
target_link_libraries(sumcheck_bench PRIVATE sumcheck Catch2::Catch2WithMain)
target_compile_options(sumcheck_bench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       --use_fast_math
                       --relocatable-device-code=true
                       >)

add_executable(gpu-benchmarks src/ulvt/finite_fields/tests/profiling/gpu_benchmarks.cu)
target_link_libraries(gpu-benchmarks PRIVATE ulvt_gpu nvbench::nvbench)
target_compile_options(gpu-benchmarks PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       --use_fast_math
                       --relocatable-device-code=true
                       >)

add_executable(prime_fields_gpu src/ulvt/prime_field_sumcheck/test_sumcheck.cu)
target_link_libraries(prime_fields_gpu PRIVATE prime_field_sumcheck Catch2::Catch2WithMain)
target_compile_options(prime_fields_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       --use_fast_math
                       --relocatable-device-code=true
                       >)

add_executable(circuit_generator src/ulvt/finite_fields/circuit_generator/multiply_and_generate_circuit.cpp)
target_link_libraries(circuit_generator PRIVATE ulvt_gpu)

# Release-specific definitions
set(cuda_defs "$<$<CONFIG:Release>:STRIP_ASSERTIONS>" "$<$<CONFIG:Release>:STRIP_CUDA_CHECK>")

target_compile_definitions(ulvt_gpu PUBLIC ${cuda_defs})
target_compile_definitions(ntt_tests PUBLIC ${cuda_defs})
target_compile_definitions(finite_field_tests PUBLIC ${cuda_defs})
target_compile_definitions(sumcheck_test PUBLIC ${cuda_defs})
