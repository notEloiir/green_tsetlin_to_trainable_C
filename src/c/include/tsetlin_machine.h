/*
(Heavily) Modified code from https://github.com/cair/TsetlinMachineC
*/


#include <stdint.h>

struct TsetlinMachine {
    uint32_t num_classes;
    uint32_t threshold;
    uint32_t num_literals;
    uint32_t num_clauses;
    int8_t max_state, min_state;
    uint8_t boost_true_positive_feedback;

    uint8_t (*y_eq)(const struct TsetlinMachine *tm, const void *y, const int32_t *y_pred, uint32_t y_size);

    uint8_t predict;
    uint8_t update;

	int mid_state;
	int8_t *ta_state;  // shape: flat (num_clauses, num_literals, 2)
	int16_t *weights;  // shape: flat (num_classes, num_clauses)
	int32_t *clause_output;  // shape: (num_clauses)
	int32_t *clause_feedback;  // shape: flat (num_classes, num_clauses)
};

// Basic y_eq functions, you can implement your own
uint8_t y_eq_class_idx(const struct TsetlinMachine *tm, const void *y, const int32_t *y_pred, uint32_t y_size);  // y_size=1
uint8_t y_eq_bin_vector(const struct TsetlinMachine *tm, const void *y, const int32_t *y_pred, uint32_t y_size);  // y_size=tm->num_classes

// X shape: flat (rows, num_literals)
// y shape: flat (rows, y_size)
// y_pred shape (per row): (num_classes)

// Create a Tsetlin Machine. Number of classes corresponds to number of bits in the TM output.
struct TsetlinMachine *create_tsetlin_machine(
    uint32_t num_classes, uint32_t threshold, uint32_t num_literals, uint32_t num_clauses,
    int8_t max_state, int8_t min_state, uint8_t boost_true_positive_feedback,
    uint8_t (*y_eq)(const struct TsetlinMachine *tm, const void *y, const int32_t *y_pred, uint32_t y_size),
    uint8_t predict, uint8_t update
);

struct TsetlinMachine *load_tsetlin_machine(
    const char *filename,
    uint8_t (*y_eq)(const struct TsetlinMachine *tm, const void *y, const int32_t *y_pred, uint32_t y_size)
);

// Deallocate all memory.
void free_tsetlin_machine(struct TsetlinMachine *tm);

// Train on a single data point.
void tm_update(struct TsetlinMachine *tm, uint8_t *X, int32_t *y, float s);

// Inference on a single data point.
// Writes to the result array y_pred of size (num_classes) elements in range [-threshold, threshold].
void tm_score(struct TsetlinMachine *tm, uint8_t *X, int32_t *y_pred);

void eval_model(struct TsetlinMachine *tm, uint8_t *X, int32_t *y, uint32_t rows, uint32_t X_cols, uint32_t y_cols);

int8_t tm_get_state(struct TsetlinMachine *tm, uint32_t clause_id, uint32_t literal_id, uint8_t automaton_type);

int16_t tm_get_weight(struct TsetlinMachine *tm, uint32_t class_id, uint32_t clause_id);

