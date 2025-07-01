#include <cstdint>
#include <iostream>

#include "../../finite_fields/circuit_generator/unrolled/binary_tower_unrolled.cuh"
#include "../../utils/bitslicing.cuh"
#include "../utils/constants.hpp"
#include "core.cuh"

__host__ __device__ void evaluate_composition_on_batch_row(
	const uint32_t* first_batch_of_row,
	uint32_t* batch_composition_destination,
	const uint32_t composition_size,
	const uint32_t original_evals_per_col
) {
	memcpy(batch_composition_destination, first_batch_of_row, BITS_WIDTH * sizeof(uint32_t));

	for (int operand_in_composition = 1; operand_in_composition < composition_size; ++operand_in_composition) {
		const uint32_t* nth_batch_of_row =
			first_batch_of_row + operand_in_composition * original_evals_per_col * INTS_PER_VALUE;

		multiply_unrolled<TOWER_HEIGHT>(batch_composition_destination, nth_batch_of_row, batch_composition_destination);
	}
}

__host__ __device__ void fold_batch(
	const uint32_t lower_batch[BITS_WIDTH],
	const uint32_t upper_batch[BITS_WIDTH],
	uint32_t dst_batch[BITS_WIDTH],
	const uint32_t coefficient[BITS_WIDTH],
	const bool is_interpolation
) {
	uint32_t xor_of_halves[BITS_WIDTH];

	for (int i = 0; i < BITS_WIDTH; ++i) {
		xor_of_halves[i] = lower_batch[i] ^ upper_batch[i];
	}

	uint32_t product[BITS_WIDTH];
	memset(product, 0, BITS_WIDTH * sizeof(uint32_t));

	// Multiply chunk-wise based on field height of coefficient
	// For random challenges this will be the full 7
	// For interpolation points this will be no more than 2

	if (is_interpolation) {
		for (int i = 0; i < BITS_WIDTH; i += INTERPOLATION_BITS_WIDTH) {
			multiply_unrolled<INTERPOLATION_TOWER_HEIGHT>(xor_of_halves + i, coefficient, product + i);
		}
	} else {
		multiply_unrolled<TOWER_HEIGHT>(xor_of_halves, coefficient, product);
	}

	for (int i = 0; i < BITS_WIDTH; ++i) {
		dst_batch[i] = lower_batch[i] ^ product[i];
	}
}

/**********************************************************************
* multiply_unrolled2_fast : same result as multiply_unrolled<2>       *
*                                                                 ... *
**********************************************************************/
__device__ __host__ __forceinline__
    void multiply_unrolled2_fast(const uint32_t  a[4],  
                                const uint32_t  preA[5],
                                const uint32_t  b[4],
                                const uint32_t  preB[5],
                                uint32_t        dst[4]){
	uint32_t v1  = a[3] & b[3];
	uint32_t v4  = v1;
	uint32_t v5  = preA[3];       
	uint32_t v7  = preB[3];       
	v1         ^= a[2] & b[2];
	uint32_t v9  = v1;
	v4         ^= v1 ^ (v5 & v7); 
	uint32_t v10 = v4;
	uint32_t v11 = v9 ^ v4;

	uint32_t v12 = preA[0];           
	uint32_t v14 = preA[1];           
	uint32_t v16 = preB[0];           
	uint32_t v18 = preB[1];           

	uint32_t v20 = a[1] & b[1];
	v4          ^= v20;
	uint32_t v21 = preA[2];           
	uint32_t v22 = preB[2];           
	v20         ^= a[0] & b[0];
	v9          ^= v20;
	v4          ^= v20 ^ (v21 & v22); 
	uint32_t v23 = v9;
	uint32_t v24 = v4;
	v10         ^= v9;
	v11         ^= v4;

	uint32_t v25 = v14 & v18;
	v11         ^= v25;
	uint32_t v26 = preA[4];           
	uint32_t v27 = preB[4];           
	v25         ^= v12 & v16;
	v10         ^= v25;
	v11         ^= v25 ^ (v26 & v27);

	dst[0] = v23;
	dst[1] = v24;
	dst[2] = v10;
	dst[3] = v11;
    
}

__host__ __device__ void fold_batch_interpolation(
    const uint32_t lower_batch[BITS_WIDTH],
    const uint32_t xor_of_halves[BITS_WIDTH],
    const uint32_t *pre_computes_param_A,
    uint32_t dst_batch[BITS_WIDTH],
    const uint32_t coefficient[BITS_WIDTH]
) {

    uint32_t product[BITS_WIDTH];
    memset(product, 0, BITS_WIDTH * sizeof(uint32_t));

	uint32_t pre_computes_param_B[PRE_COMPUTES_HEIGHT_2_SIZE];
	precompute_param_height_2(coefficient, pre_computes_param_B);


    // Multiply chunk-wise based on field height of coefficient
    // For interpolation points this will be no more than 2

	for (int i = 0, pre_compute_A_width = 0; i < BITS_WIDTH; i += INTERPOLATION_BITS_WIDTH, pre_compute_A_width += PRE_COMPUTES_HEIGHT_2_SIZE) {
        const uint32_t *a     = xor_of_halves + i;
        const uint32_t *current_precomputes_A  = pre_computes_param_A + pre_compute_A_width;
		multiply_unrolled2_fast(a, current_precomputes_A, coefficient, pre_computes_param_B, product + i);
	}

    for (int i = 0; i < BITS_WIDTH; ++i) {
        dst_batch[i] = lower_batch[i] ^ product[i];
    }
}


void fold_small(
	const uint32_t source[BITS_WIDTH],
	uint32_t destination[BITS_WIDTH],
	const uint32_t coefficient[BITS_WIDTH],
	const uint32_t list_len
) {
	uint32_t half_len = list_len / 2;

	uint32_t batch_to_be_multiplied[BITS_WIDTH];

	memcpy(batch_to_be_multiplied, source, BITS_WIDTH * sizeof(uint32_t));

	for (int i = 0; i < BITS_WIDTH; ++i) {
		batch_to_be_multiplied[i] >>= half_len;  // Move the upper half into the lower half of this operand
		batch_to_be_multiplied[i] ^= source[i];  // Add two halves before multiplying
	}

	uint32_t product[BITS_WIDTH];

	multiply_unrolled<TOWER_HEIGHT>(batch_to_be_multiplied, coefficient, product);

	for (int i = 0; i < BITS_WIDTH; ++i) {
		destination[i] = source[i] ^ product[i];
	}
}

__host__ __device__ void compute_sum(
	uint32_t sum[INTS_PER_VALUE],
	uint32_t bitsliced_batch[BITS_WIDTH],
	const uint32_t num_eval_points_being_summed_unpadded
) {
	BitsliceUtils<BITS_WIDTH>::bitslice_untranspose(bitsliced_batch);

	memset(sum, 0, INTS_PER_VALUE * sizeof(uint32_t));

	for (uint32_t i = 0; i < min(BITS_WIDTH, INTS_PER_VALUE * num_eval_points_being_summed_unpadded); ++i) {
		sum[i % INTS_PER_VALUE] ^= bitsliced_batch[i];
	}
}