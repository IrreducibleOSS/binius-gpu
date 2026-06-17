# Additive NTT over binary tower fields.
add_library(ulvt_ntt STATIC nttconf.cu)
target_link_libraries(ulvt_ntt PUBLIC ulvt_common)

add_executable(ntt_tests
    tests/md5.cu
    tests/test_ntt.cu
)
target_link_libraries(ntt_tests
    PRIVATE ulvt_ntt ulvt_utils ulvt_cuda_opts Catch2::Catch2WithMain)
