#pragma once

#include "tsetlin_machine.h"


// --- output_activation ---
// The raw output of a Tsetlin Machine are just summed up votes (tm->votes), of shape (num_classes)
// This function translates votes into a desirable format of any type (void *)

void oa_class_idx(const struct TsetlinMachine *tm, void *y_pred);  // y_size = 1
void oa_bin_vector(const struct TsetlinMachine *tm, void *y_pred);  // y_size = tm->num_classes

// --- calculate_feedback ---
// Calculate clause-class feedback

void feedback_class_idx(const struct TsetlinMachine *tm, const void *y, uint32_t clause_id);  // y_size = 1
void feedback_bin_vector(const struct TsetlinMachine *tm, const void *y, uint32_t clause_id);  // y_size = tm->num_classes
