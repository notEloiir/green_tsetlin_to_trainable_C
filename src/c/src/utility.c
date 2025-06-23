#include "utility.h"


int32_t clip(const int32_t x, const int32_t threshold) {
    if (x > threshold) return threshold;
    else if (x < -threshold) return -threshold;
    return x;
}


uint32_t next_cache_line_mult(uint32_t x) {
	// TODO: try getting cache line size dynamically

	// Easiest, system and architecture-agnostic estimate
	// Worst case scenario wastes some kB
	uint8_t cache_line_size = 64;

	return x + cache_line_size - 1 - (x + cache_line_size - 1) % cache_line_size;
}
