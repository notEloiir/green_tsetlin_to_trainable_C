#include "utility.h"


int32_t clip(const int32_t x, const int32_t threshold) {
    if (x > threshold) return threshold;
    else if (x < -threshold) return -threshold;
    return x;
}
