#pragma once

#include <stdint.h>


// --- Stateless (Sparse) Tsetlin Machine ---

struct TANode {
	uint32_t ta_id;
    struct TANode *next;
};
void ta_stateless_insert(struct TANode **head_ptr, struct TANode *prev, uint32_t ta_id, struct TANode **result);
void ta_stateless_remove(struct TANode **head_ptr, struct TANode *prev, struct TANode **result);

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


// X shape: flat (rows, num_literals) of uint8_t (as booleans)
// y shape: flat (rows, y_size) with element size (y_element_size) of any type (void *)
// y_pred shape: flat (rows, num_classes) with element size (y_element_size) of any type (void *)

// Create a Tsetlin Machine
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

// Deallocate all memory
void sltm_free(struct StatelessTsetlinMachine *sltm);

// Inference
// Writes to the result array y_pred of size (rows * sltm->y_size) and element size sltm->y_element_size (same as y)
void sltm_predict(struct StatelessTsetlinMachine *sltm, const uint8_t *X, void *y_pred, uint32_t rows);

// Simple accuracy evaluation
void sltm_evaluate(struct StatelessTsetlinMachine *sltm, const uint8_t *X, const void *y, uint32_t rows);


// --- y_eq ---
// Since y and y_pred are of any type, this function determines whether y == y_pred

// Basic y_eq function comparing raw memory using memcmp
// Works with any trivial types
uint8_t sltm_y_eq_generic(const struct StatelessTsetlinMachine *sltm, const void *y, const void *y_pred);


// --- output_activation ---
// The raw output of a Tsetlin Machine are just summed up votes (sltm->votes), of shape (num_classes)
// This function translates votes into a desirable format of any type (void *)

void sltm_oa_class_idx(const struct StatelessTsetlinMachine *sltm, const void *y_pred);  // y_size = 1
void sltm_oa_bin_vector(const struct StatelessTsetlinMachine *sltm, const void *y_pred);  // y_size = tm->num_classes

void sltm_set_output_activation(
    struct StatelessTsetlinMachine *sltm,
    void (*output_activation)(const struct StatelessTsetlinMachine *sltm, const void *y_pred)
);
