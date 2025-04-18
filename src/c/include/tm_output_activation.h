#pragma once

#include "tsetlin_machine.h"

// Basic output_activation funtions, you can implement your own
void oa_class_idx(const struct TsetlinMachine *tm, const int32_t *votes, void *y_pred);  // y_size = 1
void oa_bin_vector(const struct TsetlinMachine *tm, const int32_t *votes, void *y_pred);  // y_size = tm->num_classes

// Basic output_activation_pseudograd funtions, you can implement your own
void oa_class_idx_pseudograd(const struct TsetlinMachine *tm, const void *y, const int32_t *votes, int8_t *grad);  // y_size = 1
void oa_binary_vector_pseudograd(const struct TsetlinMachine *tm, const void *y, const int32_t *votes, int8_t *grad);  // y_size = tm->num_classes
