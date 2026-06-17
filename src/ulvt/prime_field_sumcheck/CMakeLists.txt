# Prime-field (QM31) sumcheck.
add_library(prime_field_sumcheck STATIC core/kernels.cu)
target_link_libraries(prime_field_sumcheck PUBLIC ulvt_common)

add_executable(prime_fields_gpu test_sumcheck.cu)
target_link_libraries(prime_fields_gpu
    PRIVATE prime_field_sumcheck ulvt_cuda_opts Catch2::Catch2WithMain)
