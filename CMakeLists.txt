# Version stuff
cmake_minimum_required(VERSION 3.27)
project(ulvt_gpu VERSION 0.1 LANGUAGES CXX CUDA)

set_property(GLOBAL PROPERTY CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add 3rdparty stuff
add_subdirectory(third-party/Catch2)
add_subdirectory(third-party/nvbench)


# sources and headers
set(ntt_lib_srcs "./src/ulvt/ntt/nttconf.cu")
set(ntt_lib_headers "./src/ulvt/ntt/gpuntt.cuh" "./src/ulvt/ntt/nttconf.cuh" "./src/ulvt/ntt/additive_ntt.cuh")
file(GLOB ntt_test_srcs "./src/ulvt/ntt/tests/*.cu")

file(GLOB circuit_util_srcs "./src/ulvt/finite_fields/circuit_generator/utils/*.cpp")
file(GLOB circuit_util_headers "./src/ulvt/finite_fields/circuit_generator/utils/*.hpp")

file(GLOB unrolled_srcs "./src/ulvt/finite_fields/circuit_generator/unrolled/*.cu")
file(GLOB unrolled_headers "./src/ulvt/finite_fields/circuit_generator/unrolled/*.cuh")

add_library(unrolled STATIC ${unrolled_srcs} ${unrolled_headers})
set_target_properties(unrolled PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

file(GLOB finite_field_kernel_srcs "./src/ulvt/finite_fields/tests/profiling/kernels/*.cu")
file(GLOB finite_field_kernel_headers "./src/ulvt/finite_fields/tests/profiling/kernels/*.cuh")

file(GLOB sumcheck_test_src "./src/ulvt/sumcheck/test/test.cu")

file(GLOB sumcheck_test_verifier_src "./src/ulvt/sumcheck/test/verifier.cu")
file(GLOB sumcheck_test_verifier_header "./src/ulvt/sumcheck/test/verifier.cuh")

file(GLOB sumcheck_test_utils_srcs "./src/ulvt/sumcheck/test/utils/*.cu")
file(GLOB sumcheck_test_utils_headers "./src/ulvt/sumcheck/test/utils/*.cuh")

file(GLOB sumcheck_test_kernel_srcs "./src/ulvt/sumcheck/test/kernel/**.cu")
file(GLOB sumcheck_test_kernel_headers "./src/ulvt/sumcheck/test/kernel/**.cuh")

file(GLOB sumcheck_bench_srcs "./src/ulvt/sumcheck/bench/benchmark.cu")

file(GLOB sumcheck_utils_headers_cpp "./src/ulvt/sumcheck/utils/*.hpp")

file(GLOB sumcheck_protocol_headers "./src/ulvt/sumcheck/*.cuh")

file(GLOB sumcheck_core_srcs "./src/ulvt/sumcheck/core/*.cu")
file(GLOB sumcheck_core_headers "./src/ulvt/sumcheck/core/*.cuh")

file(GLOB bitslicing_srcs "./src/ulvt/utils/bitslicing.cu")
file(GLOB bitslicing_headers "./src/ulvt/utils/bitslicing.cuh")


add_library(sumcheck STATIC 
${sumcheck_utils_headers_cpp}
${sumcheck_utils_headers}
${sumcheck_protocol_headers}
${sumcheck_core_srcs}
${sumcheck_core_headers}
${bitslicing_srcs}
${bitslicing_headers}
${tower_mul_unbitsliced}
${sumcheck_test_utils_srcs}
${sumcheck_test_utils_headers}
)

add_library(prime_field_sumcheck STATIC 
    src/ulvt/prime_field_sumcheck/sumcheck.cuh
    src/ulvt/prime_field_sumcheck/core/kernels.cu
    src/ulvt/prime_field_sumcheck/core/kernels.cuh
    src/ulvt/prime_field_sumcheck/utils/interpolate.hpp
)

set_target_properties(sumcheck PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(prime_field_sumcheck PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


target_include_directories(sumcheck PUBLIC "./src")
target_link_libraries(sumcheck PRIVATE unrolled)


set(utils_lib_srcs "./src/ulvt/utils/common.cu")
set(utils_lib_headers "./src/ulvt/utils/common.cuh" )


set(finite_field_headers "./src/ulvt/finite_fields/baby_bear.cuh"
                       "./src/ulvt/finite_fields/risc0_baby_bear.h"
                       "./src/ulvt/finite_fields/binary_tower.cuh"
                       "./src/ulvt/finite_fields/binary_tower_simd.cuh"
)

file(GLOB finite_field_test_srcs  "./src/ulvt/finite_fields/tests/*.cu")

add_library(ulvt_gpu STATIC ${ntt_lib_srcs} ${ntt_lib_headers}
                            ${utils_lib_srcs} ${utils_lib_headers}
                            ${finite_field_headers}
                            ${finite_field_kernel_srcs}
                            ${finite_field_kernel_headers}
                            ${circuit_util_srcs}
                            ${circuit_util_headers}
            )

set_target_properties(ulvt_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(ulvt_gpu PUBLIC "./src")
target_link_libraries(ulvt_gpu PRIVATE unrolled)


# Unit tests
add_executable(ntt_tests ${ntt_test_srcs})
target_link_libraries(ntt_tests PRIVATE ulvt_gpu Catch2::Catch2WithMain)
target_compile_options(ntt_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       --use_fast_math
                       --relocatable-device-code=true
                       >)

set_target_properties(ntt_tests PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(finite_field_tests ${finite_field_test_srcs} )
target_link_libraries(finite_field_tests PRIVATE ulvt_gpu Catch2::Catch2WithMain)
target_compile_options(finite_field_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       --use_fast_math
                       --relocatable-device-code=true
                       >)
set_target_properties(finite_field_tests PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(sumcheck_test ${sumcheck_test_src}
${sumcheck_test_kernel_srcs}
${sumcheck_test_kernel_headers}
# ${sumcheck_test_utils_srcs}
# ${sumcheck_test_utils_headers}
${sumcheck_test_verifier_src}
${sumcheck_test_verifier_header}
)

target_link_libraries(sumcheck_test PRIVATE sumcheck Catch2::Catch2WithMain)
target_compile_options(sumcheck_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       --use_fast_math
                       --relocatable-device-code=true
                       >)
set_target_properties(sumcheck_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(sumcheck_bench ${sumcheck_bench_srcs} )
target_link_libraries(sumcheck_bench PRIVATE sumcheck Catch2::Catch2WithMain)
target_compile_options(sumcheck_bench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       --use_fast_math
                       --relocatable-device-code=true
                       >)
set_target_properties(sumcheck_bench PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

# GPU Benchmarks
file(GLOB gpu_benchmark_srcs "./src/ulvt/finite_fields/tests/profiling/gpu_benchmarks.cu")

add_executable(gpu-benchmarks ${gpu_benchmark_srcs})
target_link_libraries(gpu-benchmarks PRIVATE ulvt_gpu nvbench::nvbench)
target_compile_options(gpu-benchmarks PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       --use_fast_math
                       --relocatable-device-code=true
                       >)

set_target_properties(gpu-benchmarks PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

# GPU Benchmarks Prime Fields
file(GLOB gpu_prime_srcs "./src/ulvt/prime_field_sumcheck/test_sumcheck.cu")

add_executable(prime_fields_gpu ${gpu_prime_srcs})
target_link_libraries(prime_fields_gpu PRIVATE prime_field_sumcheck Catch2::Catch2WithMain)
target_compile_options(prime_fields_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       --use_fast_math
                       --relocatable-device-code=true
                       >)

set_target_properties(prime_fields_gpu PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

# Circuit Generator
file(GLOB circuit_srcs "./src/ulvt/finite_fields/circuit_generator/multiply_and_generate_circuit.cpp")

add_executable(circuit_generator ${circuit_srcs})
target_link_libraries(circuit_generator PRIVATE ulvt_gpu)


# Release specific definitions
set(cuda_defs "$<$<CONFIG:Release>:STRIP_ASSERTIONS>" "$<$<CONFIG:Release>:STRIP_CUDA_CHECK>")

target_compile_definitions(ulvt_gpu PUBLIC ${cuda_defs} )
target_compile_definitions(ntt_tests PUBLIC ${cuda_defs})
target_compile_definitions(finite_field_tests PUBLIC ${cuda_defs})
target_compile_definitions(sumcheck_test PUBLIC ${cuda_defs})

