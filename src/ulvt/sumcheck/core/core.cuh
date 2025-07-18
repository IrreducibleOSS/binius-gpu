#pragma once
#include <cstdint>

#include "../utils/constants.hpp"

#define MAX_INTERPOLATING_POINTS 10


extern __constant__ uint8_t  device_matrix_rows_height_2[MAX_INTERPOLATING_POINTS][4];
extern __constant__ uint32_t device_matrix_rows_height_7[128][4];

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

__host__ __device__ void pre_calculate_lookup_table_4bit( const uint32_t input[4],
                                                    uint32_t       output[16] );


__device__ void fold_batch_interpolated_height_2_via_precomputes(
    const uint32_t lower_batch[BITS_WIDTH],
     const uint32_t xor_of_halves[BITS_WIDTH],
    //  const uint32_t pre_computes_look_ups[32][16],
	uint32_t dst_batch[BITS_WIDTH],
	const uint32_t interpolation_point
);