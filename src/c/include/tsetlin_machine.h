#pragma once

#include <stdint.h>


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
    void (*output_activation)(const struct TsetlinMachine *tm, void *y_pred);
    void (*output_activation_pseudograd)(const struct TsetlinMachine *tm, const void *y, int8_t *pseudograd);

	int8_t mid_state;
    float s_inv, s_min1_inv;
	int8_t *ta_state;  // shape: flat (num_clauses, num_literals, 2)
	int16_t *weights;  // shape: flat (num_clauses, num_classes)
	uint8_t *clause_output;  // shape: (num_clauses)
    int8_t *pseudograd;  // shape: flat (num_clauses, num_classes)
    int32_t *votes;  // shape: (num_classes)
};


// X shape: flat (rows, num_literals) of uint8_t (as booleans)
// y shape: flat (rows, y_size) with element size (y_element_size) of any type (void *)
// y_pred shape: flat (rows, num_classes) with element size (y_element_size) of any type (void *)

// Create a Tsetlin Machine
struct TsetlinMachine *tm_create(
    uint32_t num_classes, uint32_t threshold, uint32_t num_literals, uint32_t num_clauses,
    int8_t max_state, int8_t min_state, uint8_t boost_true_positive_feedback, 
    uint32_t y_size, uint32_t y_element_size, float s
);

// Load Tsetlin Machine from a bin file
struct TsetlinMachine *tm_load(
    const char *filename, uint32_t y_size, uint32_t y_element_size
);

// Deallocate all memory
void tm_free(struct TsetlinMachine *tm);

// Train on a single data point
void tm_update(struct TsetlinMachine *tm, uint8_t *X, void *y);

// Inference
// Writes to the result array y_pred of size (rows * tm->y_size) and element size tm->y_element_size
void tm_score(struct TsetlinMachine *tm, uint8_t *X, void *y_pred, uint32_t rows);

// Evaluation
void tm_eval(struct TsetlinMachine *tm, uint8_t *X, void *y, uint32_t rows);


// --- y_eq ---
// Since y and y_pred are of any type, this function determines whether y == y_pred

// Basic y_eq function comparing raw memory using memcmp
// Works with any trivial types
uint8_t y_eq_generic(const struct TsetlinMachine *tm, const void *y, const void *y_pred);


// --- output_activation ---
// The raw output of a Tsetlin Machine are just summed up votes (tm->votes), of shape (num_classes)
// This function translates votes into a desirable format of any type (void *)

// Defined in tm_output_activation.h

void tm_set_output_activation(
    struct TsetlinMachine *tm,
    void (*output_activation)(const struct TsetlinMachine *tm, void *y_pred)
);


// --- output_activation_pseudograd ---
// The pseudo gradient to output_activation function
// This function, based on votes, decides the vector to minimize cost function
// (Since output_activation doesn't have to be differentiable, can be a heuristic, this is a pseudo gradient)

// Defined in tm_output_activation.h

void tm_set_output_activation_pseudograd(
    struct TsetlinMachine *tm,
    void (*output_activation_pseudograd)(const struct TsetlinMachine *tm, const void *y, int8_t *pseudograd)
);


// --- Getters ---

int8_t tm_get_state(struct TsetlinMachine *tm, uint32_t clause_id, uint32_t literal_id, uint8_t automaton_type);

int16_t tm_get_weight(struct TsetlinMachine *tm, uint32_t class_id, uint32_t clause_id);

