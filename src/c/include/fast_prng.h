#pragma once

#include <stdint.h>


// A very fast, simple PRNG using xorshift32
struct FastPRNG {
	uint32_t state;
};

// Set the seed for the PRNG
void prng_seed(struct FastPRNG* prng, uint32_t seed);

// Generate a pseudo-random uint32_t [0, UINT32_MAX]
uint32_t prng_next_uint32(struct FastPRNG* prng);

// Generate a pseudo-random float [0, 1)
float prng_next_float(struct FastPRNG* prng);
