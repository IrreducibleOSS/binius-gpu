# Binary-tower sumcheck.
add_library(sumcheck STATIC
    core/core.cu
    core/kernels.cu
)
target_link_libraries(sumcheck PUBLIC ulvt_common PRIVATE unrolled)

add_executable(sumcheck_test
    test/test.cu
    test/kernel/verifier_kernel.cu
    test/utils/bigints.cu
    test/utils/tower_7_mul.cu
    test/verifier.cu
)
target_link_libraries(sumcheck_test PRIVATE sumcheck ulvt_cuda_opts Catch2::Catch2WithMain)

add_executable(sumcheck_bench bench/benchmark.cu)
target_link_libraries(sumcheck_bench PRIVATE sumcheck ulvt_cuda_opts Catch2::Catch2WithMain)
