#pragma once

#include <stdint.h>
#include "fast_prng.h"
#include "stateless_tsetlin_machine.h"


// --- Sparse Tsetlin Machine ---

struct TAStateNode {
	uint32_t ta_id;
    int8_t ta_state;
    uint8_t has_state;  // 0b11111111 = has state, 0b00000000 = no state, can be used as bit mask
    struct TAStateNode *next;
};
void ta_state_insert(struct TAStateNode **head_ptr, struct TAStateNode *prev, uint32_t ta_id, uint8_t ta_state, struct TAStateNode **result);
void ta_state_remove(struct TAStateNode **head_ptr, struct TAStateNode *prev, struct TAStateNode **result);

struct SparseTsetlinMachine {
    uint32_t num_classes;
    uint32_t threshold;
    uint32_t num_literals;
    uint32_t num_clauses;
    int8_t max_state, min_state, sparse_init_state;
    uint8_t boost_true_positive_feedback;
    float s;

    uint32_t y_size, y_element_size;
    uint8_t (*y_eq)(const struct SparseTsetlinMachine *stm, const void *y, const void *y_pred);
    void (*output_activation)(const struct SparseTsetlinMachine *stm, const void *y_pred);
    void (*calculate_feedback)(struct SparseTsetlinMachine *stm, const uint8_t *X, const void *y);

    int8_t mid_state;
    float s_inv, s_min1_inv;
    struct TAStateNode **ta_state;  // shape: (num_clauses) linked list pointers
    struct TANode **active_literals;  // shape: (num_classes) linked list pointers
    int16_t *weights;  // shape: flat (num_clauses, num_classes)
    uint8_t *clause_output;  // shape: (num_clauses)
    int32_t *votes;  // shape: (num_classes)

    struct FastPRNG rng;
};


// X shape: flat (rows, num_literals) of uint8_t (as booleans)
// y shape: flat (rows, y_size) with element size (y_element_size) of any type (void *)
// y_pred shape: flat (rows, num_classes) with element size (y_element_size) of any type (void *)

// Create a Tsetlin Machine
struct SparseTsetlinMachine *stm_create(
    uint32_t num_classes, uint32_t threshold, uint32_t num_literals, uint32_t num_clauses,
    int8_t max_state, int8_t min_state, uint8_t boost_true_positive_feedback, 
    uint32_t y_size, uint32_t y_element_size, float s, uint32_t seed
);

// Load Tsetlin Machine from a bin file
struct SparseTsetlinMachine *stm_load_dense(
    const char *filename, uint32_t y_size, uint32_t y_element_size
);

// Save Tsetlin Machine to a bin file
void stm_save(const struct SparseTsetlinMachine *stm, const char *filename);

// Deallocate all memory
void stm_free(struct SparseTsetlinMachine *stm);

// Train
void stm_train(struct SparseTsetlinMachine *stm, const uint8_t *X, const void *y, uint32_t rows, uint32_t epochs);

// Inference
// Writes to the result array y_pred of size (rows * stm->y_size) and element size stm->y_element_size (same as y)
void stm_predict(struct SparseTsetlinMachine *stm, const uint8_t *X, void *y_pred, uint32_t rows);

// Simple accuracy evaluation
void stm_evaluate(struct SparseTsetlinMachine *stm, const uint8_t *X, const void *y, uint32_t rows);


// --- y_eq ---
// Since y and y_pred are of any type, this function determines whether y == y_pred

// Basic y_eq function comparing raw memory using memcmp
// Works with any trivial types
uint8_t stm_y_eq_generic(const struct SparseTsetlinMachine *stm, const void *y, const void *y_pred);


// --- output_activation ---
// The raw output of a Tsetlin Machine are just summed up votes (stm->votes), of shape (num_classes)
// This function translates votes into a desirable format of any type (void *)

void stm_oa_class_idx(const struct SparseTsetlinMachine *stm, const void *y_pred);  // y_size = 1
void stm_oa_bin_vector(const struct SparseTsetlinMachine *stm, const void *y_pred);  // y_size = tm->num_classes

void stm_set_output_activation(
    struct SparseTsetlinMachine *stm,
    void (*output_activation)(const struct SparseTsetlinMachine *stm, const void *y_pred)
);


// --- calculate_feedback ---
// Calculate clause-class feedback

void stm_feedback_class_idx(struct SparseTsetlinMachine *stm, const uint8_t *X, const void *y);  // y_size = 1
void stm_feedback_bin_vector(struct SparseTsetlinMachine *stm, const uint8_t *X, const void *y);  // y_size = tm->num_classes

// Internal component of feedback functions, included in header if you want to create your own
void stm_apply_feedback(struct SparseTsetlinMachine *stm, uint32_t clause_id, uint32_t class_id, uint8_t is_class_positive, const uint8_t *X);

void stm_set_calculate_feedback(
    struct SparseTsetlinMachine *stm,
    void (*calculate_feedback)(struct SparseTsetlinMachine *stm, const uint8_t *X, const void *y)
);

