#pragma once

#include <stdint.h>


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
    void (*output_activation)(const struct TsetlinMachine *tm, const int32_t *votes, void *y_pred);
    void (*output_activation_pseudograd)(const struct TsetlinMachine *tm, const void *y, const int32_t *votes, int8_t *grad);

	int mid_state;
	int8_t *ta_state;  // shape: flat (num_clauses, num_literals, 2)
	int16_t *weights;  // shape: flat (num_clauses, num_classes)
	int32_t *clause_output;  // shape: (num_clauses)
    int8_t *pseudograd;  // shape: flat (num_clauses, num_classes)
};

// Basic y_eq function, you can implement your own
uint8_t y_eq_generic(const struct TsetlinMachine *tm, const void *y, const void *y_pred);

// X shape: flat (rows, num_literals)
// y shape: flat (rows, y_size)
// y_pred shape (per row): (num_classes)

// Create a Tsetlin Machine. Number of classes corresponds to number of bits in the TM output.
struct TsetlinMachine *tm_create(
    uint32_t num_classes, uint32_t threshold, uint32_t num_literals, uint32_t num_clauses,
    int8_t max_state, int8_t min_state, uint8_t boost_true_positive_feedback, 
    uint32_t y_size, uint32_t y_element_size, float s
);

struct TsetlinMachine *tm_load(
    const char *filename,
    uint32_t y_size, uint32_t y_element_size, float s
);

// Deallocate all memory.
void tm_free(struct TsetlinMachine *tm);

// Train on a single data point.
void tm_update(struct TsetlinMachine *tm, uint8_t *X, void *y);

// Inference on a single data point.
// Writes to the result array y_pred of size (num_classes) elements in range [-threshold, threshold].
void tm_score(struct TsetlinMachine *tm, uint8_t *X, int32_t *votes);

void tm_eval(struct TsetlinMachine *tm, uint8_t *X, void *y, uint32_t rows, uint32_t cols);

int8_t tm_get_state(struct TsetlinMachine *tm, uint32_t clause_id, uint32_t literal_id, uint8_t automaton_type);

int16_t tm_get_weight(struct TsetlinMachine *tm, uint32_t class_id, uint32_t clause_id);

