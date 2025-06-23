#pragma once

#include <stdint.h>
#include "fast_prng.h"


// --- Tsetlin Machine ---

struct TsetlinMachine {
    uint32_t num_classes;
    uint32_t threshold;
    uint32_t num_literals;
    uint32_t num_clauses;
    int8_t max_state, min_state;
    uint8_t boost_true_positive_feedback;
    float s;

    uint32_t y_size, y_element_size;
    uint8_t (*y_eq)(const struct TsetlinMachine *tm, const void *y, const void *y_pred);
    void (*output_activation)(const struct TsetlinMachine *tm, const void *y_pred);
    void (*calculate_feedback)(struct TsetlinMachine *tm, const uint8_t *X, const void *y);

	int8_t mid_state;
	uint8_t ta_state_padding;
	uint32_t ta_state_cols;
    float s_inv, s_min1_inv;
	int8_t *ta_state;  // shape: flat (num_clauses, num_literals, 2)
	int16_t *weights;  // shape: flat (num_clauses, num_classes)
	uint8_t *clause_output;  // shape: (num_clauses)
    int32_t *votes;  // shape: (num_classes)

    struct FastPRNG rng;
};


// X shape: flat (rows, num_literals) of uint8_t (as booleans)
// y shape: flat (rows, y_size) with element size (y_element_size) of any type (void *)
// y_pred shape: flat (rows, num_classes) with element size (y_element_size) of any type (void *)

// Create a Tsetlin Machine
struct TsetlinMachine *tm_create(
    uint32_t num_classes, uint32_t threshold, uint32_t num_literals, uint32_t num_clauses,
    int8_t max_state, int8_t min_state, uint8_t boost_true_positive_feedback, 
    uint32_t y_size, uint32_t y_element_size, float s, uint32_t seed
);

// Load Tsetlin Machine from a bin file
struct TsetlinMachine *tm_load(
    const char *filename, uint32_t y_size, uint32_t y_element_size
);

// Save Tsetlin Machine to a bin file
void tm_save(const struct TsetlinMachine *tm, const char *filename);

// Deallocate all memory
void tm_free(struct TsetlinMachine *tm);

// Train
void tm_train(struct TsetlinMachine *tm, const uint8_t *X, const void *y, uint32_t rows, uint32_t epochs);

// Inference
// Writes to the result array y_pred of size (rows * tm->y_size) and element size tm->y_element_size (same as y)
void tm_predict(struct TsetlinMachine *tm, const uint8_t *X, void *y_pred, uint32_t rows);

// Simple accuracy evaluation
void tm_evaluate(struct TsetlinMachine *tm, const uint8_t *X, const void *y, uint32_t rows);


// --- y_eq ---
// Since y and y_pred are of any type, this function determines whether y == y_pred

// Basic y_eq function comparing raw memory using memcmp
// Works with any trivial types
uint8_t tm_y_eq_generic(const struct TsetlinMachine *tm, const void *y, const void *y_pred);


// --- output_activation ---
// The raw output of a Tsetlin Machine are just summed up votes (tm->votes), of shape (num_classes)
// This function translates votes into a desirable format of any type (void *)


void tm_oa_class_idx(const struct TsetlinMachine *tm, const void *y_pred);  // y_size = 1
void tm_oa_bin_vector(const struct TsetlinMachine *tm, const void *y_pred);  // y_size = tm->num_classes

void tm_set_output_activation(
    struct TsetlinMachine *tm,
    void (*output_activation)(const struct TsetlinMachine *tm, const void *y_pred)
);


// --- calculate_feedback ---
// Calculate clause-class feedback

void tm_feedback_class_idx(struct TsetlinMachine *tm, const uint8_t *X, const void *y);  // y_size = 1
void tm_feedback_bin_vector(struct TsetlinMachine *tm, const uint8_t *X, const void *y);  // y_size = tm->num_classes

// Internal component of feedback functions, included in header if you want to create your own
void tm_apply_feedback(struct TsetlinMachine *tm, uint32_t clause_id, uint32_t class_id, uint8_t is_class_positive, const uint8_t *X);

void tm_set_calculate_feedback(
    struct TsetlinMachine *tm,
    void (*calculate_feedback)(struct TsetlinMachine *tm, const uint8_t *X, const void *y)
);

