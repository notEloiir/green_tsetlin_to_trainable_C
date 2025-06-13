#include "fast_prng.h"


// Set the seed for the PRNG
void prng_seed(struct FastPRNG* prng, uint32_t seed) {
    // Seed should not be zero for xorshift32
    prng->state = seed ? seed : 0xdeadbeef;
}

// Generate a pseudo-random uint32_t
uint32_t prng_next(struct FastPRNG* prng) {
    uint32_t x = prng->state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    prng->state = x;
    return x;
}
