#include <cstdint>

#include "../../finite_fields/qm31.cuh"

__global__ void fold_list_halves(QM31 *list, QM31 challenge,
                                 uint32_t current_col_size, uint32_t original_col_size );

__global__ void get_round_coefficients(QM31 *list, uint64_t sum_zero[4], uint64_t sum_one[4], uint64_t sum_two[4], uint32_t current_col_size, uint32_t original_col_size);

__global__ void fold_then_get_coeffs(QM31 *list, QM31 challenge, uint64_t sum_zero[4], uint64_t sum_one[4], uint64_t sum_two[4], uint32_t current_col_size, uint32_t original_col_size);