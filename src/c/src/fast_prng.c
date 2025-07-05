#include "fast_prng.h"


// Set the seed for the PRNG
void prng_seed(struct FastPRNG* prng, uint32_t seed) {
    // Seed should not be zero for xorshift32
    prng->state = seed ? seed : 0xdeadbeef;
}

// Generate a pseudo-random uint32_t
uint32_t prng_next_uint32(struct FastPRNG* prng) {
    uint32_t x = prng->state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    prng->state = x;
    return x;
}

// Generate a pseudo-random float [0, 1)
float prng_next_float(struct FastPRNG* prng) {
    uint32_t x = prng->state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    prng->state = x;

    static union {
    	uint32_t u32;
    	float f;
    } caster;

    // Reinterpret x as float
    // Zero out sign and exponent (so only random mantissa stays)
    // OR 0x3F800000 (== 1.0f)
    // - 1.0f
    caster.u32 = 0x3F800000 | x >> 9;

    return caster.f - 1.0f;
}
