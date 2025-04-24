#pragma once

#include "tsetlin_machine.h"


// --- output_activation ---
// The raw output of a Tsetlin Machine are just summed up votes (tm->votes), of shape (num_classes)
// This function translates votes into a desirable format of any type (void *)

void oa_class_idx(const struct TsetlinMachine *tm, void *y_pred);  // y_size = 1
void oa_bin_vector(const struct TsetlinMachine *tm, void *y_pred);  // y_size = tm->num_classes

// --- output_activation_pseudograd ---
// The pseudo gradient to output_activation function
// This function, based on votes, decides the vector to minimize cost function
// (Since output_activation doesn't have to be differentiable, can be a heuristic, this is a pseudo gradient)

void oa_class_idx_pseudograd(const struct TsetlinMachine *tm, const void *y, int8_t *grad);  // y_size = 1
void oa_bin_vector_pseudograd(const struct TsetlinMachine *tm, const void *y, int8_t *grad);  // y_size = tm->num_classes
