# The finite-field arithmetic itself is header-only (baby_bear, binary_tower,
# ghash, m31/cm31/qm31, ...). The targets below are the compiled pieces that
# live under this module.

# Machine-generated, fully unrolled tower-field multiply circuits.
add_library(unrolled STATIC
    circuit_generator/unrolled/binary_tower_unrolled2.cu
    circuit_generator/unrolled/binary_tower_unrolled5.cu
    circuit_generator/unrolled/binary_tower_unrolled6.cu
    circuit_generator/unrolled/binary_tower_unrolled7.cu
)
target_link_libraries(unrolled PUBLIC ulvt_common)

# String helpers used by the circuit generator and the field tests.
add_library(ulvt_circuit_utils STATIC circuit_generator/utils/utils.cpp)
target_link_libraries(ulvt_circuit_utils PUBLIC ulvt_common)

# Throughput/profiling kernels shared by finite_field_tests and gpu-benchmarks.
add_library(ulvt_ff_kernels STATIC
    tests/profiling/kernels/babybear_repeat.cu
    tests/profiling/kernels/bitsliced_repeat.cu
)
target_link_libraries(ulvt_ff_kernels PUBLIC ulvt_common unrolled)

# Tool that emits the unrolled circuits above.
add_executable(circuit_generator circuit_generator/multiply_and_generate_circuit.cpp)
target_link_libraries(circuit_generator PRIVATE ulvt_common)

add_executable(finite_field_tests
    tests/test_fanpaartower.cu
    tests/test_ghash.cu
    tests/test_kernels.cu
    tests/tests.cu
)
target_link_libraries(finite_field_tests
    PRIVATE ulvt_common unrolled ulvt_circuit_utils ulvt_ff_kernels ulvt_cuda_opts
            Catch2::Catch2WithMain)

add_executable(gpu-benchmarks tests/profiling/gpu_benchmarks.cu)
target_link_libraries(gpu-benchmarks PRIVATE ulvt_ff_kernels ulvt_cuda_opts nvbench::nvbench)

# Finite-field multiplication throughput benchmarks (CPU and GPU) on nvbench.
add_executable(field-mul-benchmarks benchmarks/bench_field_mul.cu)
target_link_libraries(field-mul-benchmarks
    PRIVATE ulvt_common ulvt_cuda_opts nvbench::nvbench)
