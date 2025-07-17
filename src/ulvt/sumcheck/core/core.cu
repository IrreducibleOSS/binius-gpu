#include <cstdint>
#include <iostream>

#include "../../finite_fields/circuit_generator/unrolled/binary_tower_unrolled.cuh"
#include "../../utils/bitslicing.cuh"
#include "../utils/constants.hpp"
#include "core.cuh"
__constant__ uint8_t device_matrix_rows_height_2[MAX_COMPOSITION_SIZE][4];

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

// Build a 16-entry lookup table for all XOR‐subsets of 4 slices.
//   input[0..3] : four 4-bit “bit-slices”
//   output[s]   : for s in [0..15], XOR of those input[idx] where bit idx of s is 1
__host__ __device__ void pre_calculate_lookup_table_4bit( const uint32_t input[4],
                                                    uint32_t       output[16] )
{

  // prev4[s] == s & (s-1)
  static const uint8_t prev4[16] = {
    0,0,0,2,0,4,4,6,0,8,8,10,8,12,12,14
  };
  // idx4[s] == ctz(s & -s)
  static const uint8_t idx4[16] = {
    0,0,1,0,2,0,1,0,3,0,1,0,2,0,1,0
  };

  output[0] = 0;
  for ( uint8_t s = 1; s < 16; ++s )
  {
    // uint8_t lsb  = s & -s;           // isolate lowest 1-bit of s
    // uint8_t prev = s ^ lsb;          // subset with that bit cleared
    // uint8_t idx  = __builtin_ctz( lsb );  // which slice to add
    uint8_t p = prev4[s];
    uint8_t i = idx4 [s];

    // output[s] = output[prev] ^ input[idx];
    output[s] = output[p] ^ input[i];
  }
}

__host__ __device__ static inline void mul_via_matrix_bitsliced_four_russians_4bit( const uint8_t rows[4],
                                                                const uint32_t X[4],
                                                                const uint32_t lookup[16],
                                                                      uint32_t Z[4] )
{

#pragma unroll
  for ( int j = 0; j < 4; ++j )
  {
    uint8_t row_mask = rows[j];  // only low 4 bits used
    Z[j] = lookup[row_mask];
  }
}

__device__ void fold_batch_interpolated_height_2_via_precomputes(
    const uint32_t lower_batch[BITS_WIDTH],
     const uint32_t xor_of_halves[BITS_WIDTH],
    //  const uint32_t pre_computes_look_ups[32][16],
	uint32_t dst_batch[BITS_WIDTH],
	const uint32_t interpolation_point
) {
	// uint32_t xor_of_halves[BITS_WIDTH];

	// for (int i = 0; i < BITS_WIDTH; ++i) {
	// 	xor_of_halves[i] = lower_batch[i] ^ upper_batch[i];
	// }

	uint32_t product[BITS_WIDTH];
	memset(product, 0, BITS_WIDTH * sizeof(uint32_t));

	// Multiply chunk-wise based on field height of coefficient
	// For random challenges this will be the full 7
	// For interpolation points this will be no more than 2

	for (int i = 0; i < BITS_WIDTH; i += INTERPOLATION_BITS_WIDTH) {
		  // 1) Build the 16-entry table for X
		uint32_t lookup[16];
		pre_calculate_lookup_table_4bit( xor_of_halves + i, lookup );

		mul_via_matrix_bitsliced_four_russians_4bit(
			device_matrix_rows_height_2[interpolation_point],
			xor_of_halves + i,
			lookup, //pre_computes_look_ups[i/INTERPOLATION_BITS_WIDTH],
			product + i
		);
		// multiply_unrolled<INTERPOLATION_TOWER_HEIGHT>(xor_of_halves + i, coefficient, product + i);
	}

	for (int i = 0; i < BITS_WIDTH; ++i) {
		dst_batch[i] = lower_batch[i] ^ product[i];
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