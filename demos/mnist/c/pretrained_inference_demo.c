#include <stdio.h>
#include <stdlib.h>

#include "tsetlin_machine.h"
#include "sparse_tsetlin_machine.h"
#include "stateless_tsetlin_machine.h"
#include "mnist_util.h"


int main() {
    srand(42);

    const char *file_path = "data/models/mnist_tm.bin";
    struct TsetlinMachine *tm = tm_load(file_path, 1, sizeof(int32_t));
    struct SparseTsetlinMachine *stm = stm_load_dense(file_path, 1, sizeof(int32_t));
    struct StatelessTsetlinMachine *sltm = sltm_load_dense(file_path, 1, sizeof(int32_t));
    if (tm == NULL || stm == NULL || sltm == NULL) {
		perror("tm_load failed");
		return 1;
	}
	
	// Print out hyperparameters
    printf("Threshold: %d\n", tm->threshold);
    printf("Features: %d\n", tm->num_literals);
    printf("Clauses: %d\n", tm->num_clauses);
    printf("Classes: %d\n", tm->num_classes);
    printf("Max state: %d\n", tm->max_state);
    printf("Min state: %d\n", tm->min_state);
    printf("Boost: %d\n", tm->boost_true_positive_feedback);
    printf("s: %f\n", tm->s);
    
    // Load in test data
	uint32_t rows = 70000;
	uint32_t cols = 784;
    uint8_t *x_data = malloc(rows * cols * sizeof(uint8_t));
    int32_t *y_data = malloc(rows * sizeof(int32_t));
    if (x_data == NULL || y_data == NULL) {
        fprintf(stderr, "Failed to allocate memory for x_data or y_data\n");
        return 1;
    }
    printf("Loading MNIST data\n");
    load_mnist_data(x_data, y_data);
    
    uint8_t *x_test = x_data + 60000 * cols;
    int32_t *y_test = y_data + 60000;

    // Evaluate models
    evaluate_models(tm, stm, sltm, x_test, y_test, 10000);

	// Clean up
    tm_free(tm);
    stm_free(stm);
    sltm_free(sltm);
    free(x_data);
    free(y_data);
    
    return 0;
}
