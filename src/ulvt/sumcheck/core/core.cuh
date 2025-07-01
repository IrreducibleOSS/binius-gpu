#pragma once
#include <cstdint>

#include "../utils/constants.hpp"

__host__ __device__ void evaluate_composition_on_batch_row(
	const uint32_t* first_batch_of_row,
	uint32_t* batch_composition_destination,
	const uint32_t composition_size,
	const uint32_t original_evals_per_col
);

__host__ __device__ void fold_batch(
	const uint32_t lower_batch[BITS_WIDTH],
	const uint32_t upper_batch[BITS_WIDTH],
	uint32_t dst_batch[BITS_WIDTH],
	const uint32_t coefficient[BITS_WIDTH],
	const bool is_interpolation
);

__host__ __device__ void fold_batch_interpolation(
    const uint32_t lower_batch[BITS_WIDTH],
    const uint32_t xor_of_halves[BITS_WIDTH],
    const uint32_t *pre_computes_param_A,
    uint32_t dst_batch[BITS_WIDTH],
    const uint32_t coefficient[BITS_WIDTH]
);

void fold_small(
	const uint32_t source[BITS_WIDTH],
	uint32_t destination[BITS_WIDTH],
	const uint32_t coefficient[BITS_WIDTH],
	const uint32_t list_len
);

__host__ __device__ void compute_sum(
	uint32_t sum[INTS_PER_VALUE],
	uint32_t bitsliced_batch[BITS_WIDTH],
	const uint32_t num_eval_points_being_summed_unpadded
);


#define PRE_COMPUTES_HEIGHT_2_SIZE 5
__device__ __host__ __forceinline__
void precompute_param_height_2(const uint32_t a[4], uint32_t out[PRE_COMPUTES_HEIGHT_2_SIZE])
{
    out[0] = a[0] ^ a[2];
    out[1] = a[1] ^ a[3];
    out[2] = a[0] ^ a[1];
    out[3] = a[2] ^ a[3];
	out[4] = out[0] ^ out[1]; 
}
