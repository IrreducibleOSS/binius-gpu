#pragma once

namespace ghash_ctmul32 {

/** Elements of GF(2^128) represented as F[X] / (X^128 + X^7 + X^2 + X + 1)
 */
class Ghash {
public:
    uint32_t limbs[4];

    // TODO: constructors taking __uint128 (limbs are little-endian)
    // TODO: operator overloads for +, -, *, =, !=, and assignment versions
    // +, - are xor
};

class GhashWide {
public:
    uint32_t limbs[8];

    // TODO: constructors taking __uint128 (limbs are little-endian)
    // TODO: operator overloads for +, -, *, =, !=, and assignment versions
    // +, - are xor
};

/* Code below from BearSSL (https://www.bearssl.org/git/BearSSL) */

/*
 * Multiplication in GF(2)[X], truncated to its low 32 bits.
 */
static inline uint32_t
bmul32(uint32_t x, uint32_t y)
{
	uint32_t x0, x1, x2, x3;
	uint32_t y0, y1, y2, y3;
	uint32_t z0, z1, z2, z3;

	x0 = x & (uint32_t)0x11111111;
	x1 = x & (uint32_t)0x22222222;
	x2 = x & (uint32_t)0x44444444;
	x3 = x & (uint32_t)0x88888888;
	y0 = y & (uint32_t)0x11111111;
	y1 = y & (uint32_t)0x22222222;
	y2 = y & (uint32_t)0x44444444;
	y3 = y & (uint32_t)0x88888888;
	z0 = (x0 * y0) ^ (x1 * y3) ^ (x2 * y2) ^ (x3 * y1);
	z1 = (x0 * y1) ^ (x1 * y0) ^ (x2 * y3) ^ (x3 * y2);
	z2 = (x0 * y2) ^ (x1 * y1) ^ (x2 * y0) ^ (x3 * y3);
	z3 = (x0 * y3) ^ (x1 * y2) ^ (x2 * y1) ^ (x3 * y0);
	z0 &= (uint32_t)0x11111111;
	z1 &= (uint32_t)0x22222222;
	z2 &= (uint32_t)0x44444444;
	z3 &= (uint32_t)0x88888888;
	return z0 | z1 | z2 | z3;
}

#if defined(__arm__)
#include <arm_acle.h>
#endif

/*
 * Bit-reverse a 32-bit word.
 */
__host__ __device__ static inline uint32_t rev32(uint32_t x) {
#if defined(__CUDA_ARCH__)
    return __brev(x);
#elif defined(__arm__)
    return __rbit(x);
#else

#define RMS(m, s)   do { \
		x = ((x & (uint32_t)(m)) << (s)) \
			| ((x >> (s)) & (uint32_t)(m)); \
	} while (0)

	RMS(0x55555555, 1);
	RMS(0x33333333, 2);
	RMS(0x0F0F0F0F, 4);
	RMS(0x00FF00FF, 8);
	return (x << 16) | (x >> 16);

#undef RMS
#endif
}

/* see bearssl_hash.h */
__host__ __device__
static void clmad(const Ghash& lhs, const Ghash& rhs, GhashWide& out) {
	/*
	 * This implementation is similar to br_ghash_ctmul() except
	 * that we have to do the multiplication twice, with the
	 * "normal" and "bit reversed" operands. Hence we end up with
	 * eighteen 32-bit multiplications instead of nine.
	 */

    const auto xw = lhs.limbs;
    const auto yw = rhs.limbs;

    /*
     * We are using Karatsuba: the 128x128 multiplication is
     * reduced to three 64x64 multiplications, hence nine
     * 32x32 multiplications. With the bit-reversal trick,
     * we have to perform 18 32x32 multiplications.
     */

    /*
     * x[0,1]*y[0,1] -> 0,1,4
     * x[2,3]*y[2,3] -> 2,3,5
     * (x[0,1]+x[2,3])*(y[0,1]+y[2,3]) -> 6,7,8
     */

    uint32_t a[18], b[18], c[18];

    a[0] = xw[0];
    a[1] = xw[1];
    a[2] = xw[2];
    a[3] = xw[3];
    a[4] = a[0] ^ a[1];
    a[5] = a[2] ^ a[3];
    a[6] = a[0] ^ a[2];
    a[7] = a[1] ^ a[3];
    a[8] = a[6] ^ a[7];

    a[ 9] = rev32(xw[0]);
    a[10] = rev32(xw[1]);
    a[11] = rev32(xw[2]);
    a[12] = rev32(xw[3]);
    a[13] = a[ 9] ^ a[10];
    a[14] = a[11] ^ a[12];
    a[15] = a[ 9] ^ a[11];
    a[16] = a[10] ^ a[12];
    a[17] = a[15] ^ a[16];

    b[0] = yw[0];
    b[1] = yw[1];
    b[2] = yw[2];
    b[3] = yw[3];
    b[4] = b[0] ^ b[1];
    b[5] = b[2] ^ b[3];
    b[6] = b[0] ^ b[2];
    b[7] = b[1] ^ b[3];
    b[8] = b[6] ^ b[7];

    b[ 9] = rev32(yw[0]);
    b[10] = rev32(yw[1]);
    b[11] = rev32(yw[2]);
    b[12] = rev32(yw[3]);
    b[13] = b[ 9] ^ b[10];
    b[14] = b[11] ^ b[12];
    b[15] = b[ 9] ^ b[11];
    b[16] = b[10] ^ b[12];
    b[17] = b[15] ^ b[16];

    for (i = 0; i < 18; i ++) {
        c[i] = bmul32(a[i], b[i]);
    }

    c[4] ^= c[0] ^ c[1];
    c[5] ^= c[2] ^ c[3];
    c[8] ^= c[6] ^ c[7];

    c[13] ^= c[ 9] ^ c[10];
    c[14] ^= c[11] ^ c[12];
    c[17] ^= c[15] ^ c[16];

    /*
     * x[0,1]*y[0,1] -> 0,9^4,1^13,10
     * x[2,3]*y[2,3] -> 2,11^5,3^14,12
     * (x[0,1]+x[2,3])*(y[0,1]+y[2,3]) -> 6,15^8,7^17,16
     */

    uint32_t vw[8];
    v[0] = c[0];
    v[1] = c[4] ^ (rev32(c[9]) >> 1);
    v[2] = c[1] ^ c[0] ^ c[2] ^ c[6] ^ (rev32(c[13]) >> 1);
    v[3] = c[4] ^ c[5] ^ c[8]
        ^ (rev32(c[10] ^ c[9] ^ c[11] ^ c[15]) >> 1);
    v[4] = c[2] ^ c[1] ^ c[3] ^ c[7]
        ^ (rev32(c[13] ^ c[14] ^ c[17]) >> 1);
    v[5] = c[5] ^ (rev32(c[11] ^ c[10] ^ c[12] ^ c[16]) >> 1);
    v[6] = c[3] ^ (rev32(c[14]) >> 1);
    v[7] = rev32(c[12]) >> 1;

    for (i = 0; i < 8; i ++) {
        out.limbs[i] ^= vw[i];
    }
}

/** Reduce modulo X^128 + X^7 + X^2 + X + 1. */
__host__ __device__
static Ghash reduce(GhashWide x) {
    auto vw = x.limbs;

    for (i = 7; i >= 4; i --) {
        auto lw = zw[i];
        vw[i - 4] ^= lw ^ (lw << 1) ^ (lw << 2) ^ (lw << 7);
        vw[i - 3] ^= (lw >> 31) ^ (lw >> 30) ^ (lw >> 25);
    }

    return Ghash {
        .limbs = { vw[0], vw[1], vw[2], vw[3] },
    };
}

}; // namespace ghash_ctmul32

