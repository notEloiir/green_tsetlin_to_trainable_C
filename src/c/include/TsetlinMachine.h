/*
(Heavily) Modified code from https://github.com/cair/TsetlinMachineC
*/


#include <stdint.h>

enum output_type {
    CLASS_IDX, BINARY_VECTOR
};

struct TsetlinMachine {
	int num_classes;
    int threshold;
    int num_literals;
    int num_clauses;
    int max_state, min_state;
    int boost_true_positive_feedback;
    enum output_type y_type;

    int predict;
    int update;

	int mid_state;
	int8_t *ta_state;  // shape: flat (num_clauses, num_literals, 2)
	int16_t *weights;  // shape: flat (num_classes, num_clauses)
	int *clause_output;  // shape: (num_clauses)
	int *clause_feedback;  // shape: flat (num_classes, num_clauses)
};

// Input shape: (num_literals)
// Output shape: (num_classes)

// Create a Tsetlin Machine. Number of classes corresponds to number of bits in the TM output.
struct TsetlinMachine *create_tsetlin_machine(
    int num_classes, int threshold, int num_literals, int num_clauses,
    int max_state, int min_state, int boost_true_positive_feedback,
    enum output_type y_type, int predict, int update
);

struct TsetlinMachine *load_tsetlin_machine(const char *filename);

// Deallocate all memory.
void free_tsetlin_machine(struct TsetlinMachine *tm);

// Train on a single data point.
void tm_update(struct TsetlinMachine *tm, uint8_t *Xi, int target, float s);

// Inference on a single data point.
// Writes to the result array of size (num_classes), elements in range [-threshold, threshold].
void tm_score(struct TsetlinMachine *tm, uint8_t *Xi, int *result);

void eval_model(struct TsetlinMachine *tm, uint8_t *X, uint32_t *y, int rows, int cols);

int tm_get_state(struct TsetlinMachine *tm, int clause_id, int literal_id, int automaton_type);

int tm_get_weight(struct TsetlinMachine *tm, int class_id, int clause_id);

