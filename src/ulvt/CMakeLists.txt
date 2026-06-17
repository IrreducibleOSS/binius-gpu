# Order respects inter-module dependencies: utils before ntt (ntt_tests links
# ulvt_utils), and finite_fields before sumcheck (sumcheck links unrolled).
add_subdirectory(utils)
add_subdirectory(ntt)
add_subdirectory(finite_fields)
add_subdirectory(sumcheck)
add_subdirectory(prime_field_sumcheck)
