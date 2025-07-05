#pragma once

#include <stdint.h>


static inline int32_t clip(const int32_t x, const int32_t threshold) {
    if (x > threshold) return threshold;
    else if (x < -threshold) return -threshold;
    return x;
}

#define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })

