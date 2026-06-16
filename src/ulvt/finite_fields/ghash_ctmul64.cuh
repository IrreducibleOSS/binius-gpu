#pragma once

namespace ghash_ctmul64 {

/** Elements of GF(2^128) represented as F[X] / (X^128 + X^7 + X^2 + X + 1)
 */
class Ghash {
public:
    uint64_t limbs[2];

    // TODO: constructors taking __uint128 (limbs are little-endian)
    // TODO: operator overloads for +, -, *, =, !=, and assignment versions
    // +, - are xor
};

class GhashWide {
public:
    uint64_t limbs[4];

    // TODO: constructors taking __uint128 (limbs are little-endian)
    // TODO: operator overloads for +, -, *, =, !=, and assignment versions
    // +, - are xor
};

/* Code below from BearSSL (https://www.bearssl.org/git/BearSSL) */

/*
 * This is the 64-bit variant of br_ghash_ctmul32(), with 64-bit operands
 * and bit reversal of 64-bit words.
 */

__host__ __device__
static inline uint64_t
bmul64(uint64_t x, uint64_t y)
{
	uint64_t x0, x1, x2, x3;
	uint64_t y0, y1, y2, y3;
	uint64_t z0, z1, z2, z3;

	x0 = x & (uint64_t)0x1111111111111111;
	x1 = x & (uint64_t)0x2222222222222222;
	x2 = x & (uint64_t)0x4444444444444444;
	x3 = x & (uint64_t)0x8888888888888888;
	y0 = y & (uint64_t)0x1111111111111111;
	y1 = y & (uint64_t)0x2222222222222222;
	y2 = y & (uint64_t)0x4444444444444444;
	y3 = y & (uint64_t)0x8888888888888888;
	z0 = (x0 * y0) ^ (x1 * y3) ^ (x2 * y2) ^ (x3 * y1);
	z1 = (x0 * y1) ^ (x1 * y0) ^ (x2 * y3) ^ (x3 * y2);
	z2 = (x0 * y2) ^ (x1 * y1) ^ (x2 * y0) ^ (x3 * y3);
	z3 = (x0 * y3) ^ (x1 * y2) ^ (x2 * y1) ^ (x3 * y0);
	z0 &= (uint64_t)0x1111111111111111;
	z1 &= (uint64_t)0x2222222222222222;
	z2 &= (uint64_t)0x4444444444444444;
	z3 &= (uint64_t)0x8888888888888888;
	return z0 | z1 | z2 | z3;
}

#if defined(__aarch64__)
#include <arm_acle.h>
#endif

__host__ __device__ static inline rev64(uint64_t x) {
#if defined(__CUDA_ARCH__)
    return __brevll(x);
#elif defined(__aarch64__)
    return __rbitll(x);
#else
#define RMS(m, s)   do { \
		x = ((x & (uint64_t)(m)) << (s)) \
			| ((x >> (s)) & (uint64_t)(m)); \
	} while (0)

	RMS(0x5555555555555555,  1);
	RMS(0x3333333333333333,  2);
	RMS(0x0F0F0F0F0F0F0F0F,  4);
	RMS(0x00FF00FF00FF00FF,  8);
	RMS(0x0000FFFF0000FFFF, 16);
	return (x << 32) | (x >> 32);

#undef RMS
#endif
}

/* see bearssl_ghash.h */
__host__ __device__
static void clmad(const Ghash& lhs, const Ghash& rhs, GhashWide& out) {
	const auto x0 = lhs.limbs[0];
	const auto x1 = lhs.limbs[1];
    const auto y0 = rhs.limbs[0];
    const auto y1 = rhs.limbs[1];

	const auto x0r = rev64(x0);
	const auto x1r = rev64(x1);
    const auto x2 = x0 ^ x1;
    const auto x2r = x0r ^ x1r;

	const auto y0r = rev64(y0);
	const auto y1r = rev64(y1);
    const auto y2 = y0 ^ y1;
    const auto y2r = y0r ^ y1r;

    const auto z0 = bmul64(x0, y0);
    const auto z1 = bmul64(x1, y1);
    auto z2 = bmul(x2, y2);

	auto z0h = bmul64(x0r, y0r);
	auto z1h = bmul64(x1r, y1r);
	auto z2h = bmul64(x2r, y2r);

	z2 ^= z0 ^ z1;
	z2h ^= z0h ^ z1h;
	z0h = rev64(z0h) >> 1;
	z1h = rev64(z1h) >> 1;
	z2h = rev64(z2h) >> 1;

	const auto v0 = z0;
	const auto v1 = z0h ^ z2;
	const auto v2 = z1 ^ z2h;
	const auto v3 = z1h;

    out.limbs[0] ^= v0;
    out.limbs[1] ^= v1;
    out.limbs[2] ^= v2;
    out.limbs[3] ^= v3;
}

/** Reduce modulo X^128 + X^7 + X^2 + X + 1. */
__host__ __device__
static Ghash reduce(GhashWide x) {
    auto v0 = x.limbs[0];
    auto v1 = x.limbs[1];
    auto v2 = x.limbs[2];
    auto v3 = x.limbs[3];

	v1 ^= v3 ^ (v3 << 1) ^ (v3 << 2) ^ (v3 << 7);
	v2 ^= (v3 >> 63) ^ (v3 >> 62) ^ (v3 >> 57);
	v0 ^= v2 ^ (v2 << 1) ^ (v2 << 2) ^ (v2 << 7);
	v1 ^= (v2 >> 63) ^ (v2 >> 62) ^ (v2 >> 57);

    return Ghash {
        .limbs = { v0, v1 },
    };
}

}; // namespace ghash_ctmul64

