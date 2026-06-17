# Shared GPU helpers (check_gpu_capabilities, bit-slicing, assert macros).
add_library(ulvt_utils STATIC common.cu)
target_link_libraries(ulvt_utils PUBLIC ulvt_common)
