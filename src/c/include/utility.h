#pragma once

#include <stdint.h>


int32_t clip(const int32_t x, const int32_t threshold);

uint32_t next_cache_line_mult(uint32_t x);

#define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })

