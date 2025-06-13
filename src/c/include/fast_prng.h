#pragma once

#include <stdint.h>


// A very fast, simple PRNG using xorshift32
struct FastPRNG {
	uint32_t state;
};

// Set the seed for the PRNG
void prng_seed(struct FastPRNG* prng, uint32_t seed);

// Generate a pseudo-random uint32_t
uint32_t prng_next(struct FastPRNG* prng);
