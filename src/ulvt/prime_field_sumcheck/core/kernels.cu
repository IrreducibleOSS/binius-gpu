#include <stdio.h>

#include "../../finite_fields/qm31.cuh"
#include "../utils/constants.hpp"
#include <cstdint>

__global__ void fold_list_halves(QM31 *list, QM31 challenge,
                                 uint32_t current_col_size, uint32_t original_col_size ) {
  const uint32_t tid =
      threadIdx.x +
      blockIdx.x * blockDim.x; // start the batch index off at the tid

  for (std::size_t col_idx = 0; col_idx < 2; ++col_idx) {
    QM31 *this_column_start = list + col_idx * original_col_size;
    for (std::size_t lower_row_idx = tid; lower_row_idx < current_col_size / 2;
         lower_row_idx += gridDim.x * blockDim.x) {

      std::size_t upper_row_idx = lower_row_idx + current_col_size / 2;
      // TODO: batch 128 rows in the same thread
      QM31 *lower_batch = this_column_start + lower_row_idx;
      QM31 *upper_batch = this_column_start + upper_row_idx;
      *lower_batch = *lower_batch + (*upper_batch - *lower_batch) * challenge;
    }
  }
}

__global__ void get_round_coefficients(QM31 *list, uint64_t sum_zero[4], uint64_t sum_one[4], uint64_t sum_two[4], uint32_t current_col_size, uint32_t original_col_size) {
  const uint32_t tid =
      threadIdx.x +
      blockIdx.x * blockDim.x;

  uint64_t this_thread_sum_zero[4] = {0, 0, 0, 0};
  uint64_t this_thread_sum_one[4] = {0, 0, 0, 0};
  uint64_t this_thread_sum_two[4] = {0, 0, 0, 0};

  for (std::size_t lower_row_idx = tid ; lower_row_idx < current_col_size / 2;
       lower_row_idx+=blockDim.x * gridDim.x) {
    std::size_t upper_row_idx = lower_row_idx + current_col_size / 2;

    QM31 *lower_batch = list + lower_row_idx;
    QM31 *upper_batch = list + upper_row_idx;

    QM31 lower = *lower_batch;
    QM31 upper = *upper_batch;

    QM31 this_row_product_zero = lower;
    QM31 this_row_product_one = upper;
    QM31 this_row_product_two = (upper - lower) + upper;

    for (std::size_t col_idx = 1; col_idx < 2; ++col_idx) {
      QM31 *this_column_start = list + col_idx * original_col_size;
      QM31 *lower_batch = this_column_start + lower_row_idx;
      QM31 *upper_batch = this_column_start + upper_row_idx;

      QM31 lower = *lower_batch;
      QM31 upper = *upper_batch;

      this_row_product_zero *= lower;
      this_row_product_one *= upper;
      this_row_product_two *= (upper - lower) + upper;
    }

    this_row_product_zero.sum_into_u64(this_thread_sum_zero);
    this_row_product_one.sum_into_u64(this_thread_sum_one);
    this_row_product_two.sum_into_u64(this_thread_sum_two);
  }
  
  for (std::size_t i = 0; i < 4; ++i) {
    atomicAdd((unsigned long long *) sum_zero + i, (unsigned long long) this_thread_sum_zero[i]);
    atomicAdd((unsigned long long *) sum_one + i, (unsigned long long) this_thread_sum_one[i]);
    atomicAdd((unsigned long long *) sum_two + i, (unsigned long long) this_thread_sum_two[i]);
  }
}

__global__ void fold_then_get_coeffs(QM31 *list, QM31 challenge, uint64_t sum_zero[4], uint64_t sum_one[4], uint64_t sum_two[4], uint32_t current_col_size, uint32_t original_col_size) {
  const uint32_t tid =
      threadIdx.x +
      blockIdx.x * blockDim.x;

  uint64_t this_thread_sum_zero[4] = {0, 0, 0, 0};
  uint64_t this_thread_sum_one[4] = {0, 0, 0, 0};
  uint64_t this_thread_sum_two[4] = {0, 0, 0, 0};

  for (std::size_t first_quarter_row_idx = tid ; first_quarter_row_idx < current_col_size / 4;
       first_quarter_row_idx+=blockDim.x * gridDim.x) {
    std::size_t second_quarter_row_idx = first_quarter_row_idx + current_col_size / 4;
    std::size_t third_quarter_row_idx = second_quarter_row_idx + current_col_size / 4;
    std::size_t fourth_quarter_row_idx = third_quarter_row_idx + current_col_size / 4;

    QM31 *first_row = list + first_quarter_row_idx;
    QM31 *second_row = list + second_quarter_row_idx;
    QM31 *third_row = list + third_quarter_row_idx;
    QM31 *fourth_row = list + fourth_quarter_row_idx;

    QM31 first = *first_row;
    QM31 second = *second_row;
    QM31 third = *third_row;
    QM31 fourth = *fourth_row;

    QM31 lower = first + challenge * (third - first);
    QM31 upper = second + challenge * (fourth - second);

    *first_row = lower;
    *second_row = upper;

    QM31 this_row_product_zero = lower;
    QM31 this_row_product_one = upper;
    QM31 this_row_product_two = (upper - lower) + upper;

    for (std::size_t col_idx = 1; col_idx < 2; ++col_idx) {
      QM31 *this_column_start = list + col_idx * original_col_size;
      
      QM31 *first_row = this_column_start + first_quarter_row_idx;
      QM31 *second_row = this_column_start + second_quarter_row_idx;
      QM31 *third_row = this_column_start + third_quarter_row_idx;
      QM31 *fourth_row = this_column_start + fourth_quarter_row_idx;

      QM31 first = *first_row;
      QM31 second = *second_row;
      QM31 third = *third_row;
      QM31 fourth = *fourth_row;

      QM31 lower = first + challenge * (third - first);
      QM31 upper = second + challenge * (fourth - second);

      *first_row = lower;
      *second_row = upper;

      this_row_product_zero *= lower;
      this_row_product_one *= upper;
      this_row_product_two *= (upper - lower) + upper;
    }

    this_row_product_zero.sum_into_u64(this_thread_sum_zero);
    this_row_product_one.sum_into_u64(this_thread_sum_one);
    this_row_product_two.sum_into_u64(this_thread_sum_two);
  }
  
  for (std::size_t i = 0; i < 4; ++i) {
    atomicAdd((unsigned long long *) sum_zero + i, (unsigned long long) this_thread_sum_zero[i]);
    atomicAdd((unsigned long long *) sum_one + i, (unsigned long long) this_thread_sum_one[i]);
    atomicAdd((unsigned long long *) sum_two + i, (unsigned long long) this_thread_sum_two[i]);
  }
}