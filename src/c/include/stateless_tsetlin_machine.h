#pragma once

#include <stdint.h>


// --- Stateless (Sparse) Tsetlin Machine ---

// Linked list node for Tsetlin Automaton ids (without state)
struct TANode {
	uint32_t ta_id;
    struct TANode *next;
};
void ta_stateless_insert(struct TANode **head_ptr, struct TANode *prev, uint32_t ta_id, struct TANode **result);
void ta_stateless_remove(struct TANode **head_ptr, struct TANode *prev, struct TANode **result);

// Don't create, modify or free this struct directly, use sltm_load_dense, stm_free, etc.
struct StatelessTsetlinMachine {
    uint32_t num_classes;
    uint32_t threshold;
    uint32_t num_literals;
    uint32_t num_clauses;
    int8_t max_state, min_state;
    uint8_t boost_true_positive_feedback;
    float s;

    uint32_t y_size, y_element_size;
    uint8_t (*y_eq)(const struct StatelessTsetlinMachine *sltm, const void *y, const void *y_pred);
    void (*output_activation)(const struct StatelessTsetlinMachine *sltm, const void *y_pred);

    int8_t mid_state;
    float s_inv, s_min1_inv;
    struct TANode **ta_state;  // shape: (num_clauses) linked list pointers
    int16_t *weights;  // shape: flat (num_clauses, num_classes)
    uint8_t *clause_output;  // shape: (num_clauses)
    int8_t *feedback;  // shape: flat (num_clauses, num_classes, 3) - clause-class feedback type strengths: 1a, 1b, 2
    int32_t *votes;  // shape: (num_classes)
};


// X shape: flat (rows, num_literals) of uint8_t (each uint8_t should be 0 or 1)
// y shape: flat (rows, y_size) with element size (y_element_size) of any type (void *)
// y_pred shape: flat (rows, num_classes) with element size (y_element_size) of any type (void *)

// Don't use this function directly, there is no point since it can't be trained
// Instead use sltm_load_dense to load (and prune) a pre-trained Tsetlin Machine from a bin file
struct StatelessTsetlinMachine *sltm_create(
    uint32_t num_classes, uint32_t threshold, uint32_t num_literals, uint32_t num_clauses,
    int8_t max_state, int8_t min_state, uint8_t boost_true_positive_feedback, 
    uint32_t y_size, uint32_t y_element_size, float s
);

// Load Tsetlin Machine from a bin file
struct StatelessTsetlinMachine *sltm_load_dense(
    const char *filename, uint32_t y_size, uint32_t y_element_size
);

// Save Tsetlin Machine to a bin file
void sltm_save(const struct StatelessTsetlinMachine *sltm, const char *filename);

// Free all allocated memory
// It also frees the TsetlinMachine struct itself
// Remember to set tm to NULL after this call
void sltm_free(struct StatelessTsetlinMachine *sltm);

// Inference
// Writes to user allocated memory y_pred
// X shape: flat (rows, num_literals) of uint8_t (each uint8_t should be 0 or 1)
// y_pred shape: flat (rows, num_classes) with element size (y_element_size) of any type (void *)
void sltm_predict(struct StatelessTsetlinMachine *sltm, const uint8_t *X, void *y_pred, uint32_t rows);

// Simple accuracy evaluation
// X shape: flat (rows, num_literals) of uint8_t (each uint8_t should be 0 or 1)
// y shape: flat (rows, y_size) with element size (y_element_size) of any type (void *)
void sltm_evaluate(struct StatelessTsetlinMachine *sltm, const uint8_t *X, const void *y, uint32_t rows);


// --- y_eq ---
// Since y and y_pred are of any type, this function determines whether y == y_pred
// You can write your own y_eq function, and set it in the TsetlinMachine struct

// Basic y_eq function comparing raw memory using memcmp
// Works with any trivial types
uint8_t sltm_y_eq_generic(const struct StatelessTsetlinMachine *sltm, const void *y, const void *y_pred);


// --- \/ DON'T USE THESE FUNCTIONS DIRECTLY \/ ---
// Unless you know what you are doing, use the stm_set_* functions to set the desired components
// or leave them as default

// --- output_activation ---
// The raw output of a Tsetlin Machine are just summed up votes (sltm->votes), of shape (num_classes)
// This function translates votes into a desirable format of any type (void *)

// Output is a class index (e.g., for classification tasks)
// y_size = 1, y_element_size = sizeof(uint32_t)
void sltm_oa_class_idx(const struct StatelessTsetlinMachine *sltm, const void *y_pred);  // y_size = 1

// Output is a binary vector of class predictions
// y_size = tm->num_classes, y_element_size = sizeof(uint8_t)
void sltm_oa_bin_vector(const struct StatelessTsetlinMachine *sltm, const void *y_pred);  // y_size = tm->num_classes

// Set the output activation function
// Provided functions are tm_oa_class_idx and tm_oa_bin_vector
// Or implement your own
// Default is tm_oa_class_idx
void sltm_set_output_activation(
    struct StatelessTsetlinMachine *sltm,
    void (*output_activation)(const struct StatelessTsetlinMachine *sltm, const void *y_pred)
);
