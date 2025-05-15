#include <stdio.h>
#include <stdlib.h>

#include "tsetlin_machine.h"
#include "sparse_tsetlin_machine.h"
#include "stateless_tsetlin_machine.h"
#include "mnist_util.h"


int main() {
    srand(42);

    uint32_t num_classes = 10;
    uint32_t threshold = 1000;
    uint32_t num_literals = 784;
    uint32_t num_clauses = 1000;
    int8_t max_state = 127;
    int8_t min_state = -127;
    uint8_t boost_true_positive_feedback = 0;
    uint32_t y_size = 1;
    uint32_t y_element_size = sizeof(int32_t);
    float s = 10.0f;

    struct TsetlinMachine *tm = tm_create(num_classes, threshold, num_literals, num_clauses,
        max_state, min_state, boost_true_positive_feedback, y_size, y_element_size, s);
    // struct SparseTsetlinMachine *stm = stm_create(num_classes, threshold, num_literals, num_clauses,
    //     max_state, min_state, boost_true_positive_feedback, y_size, y_element_size, s);
    // if (tm == NULL || stm == NULL) {
	// 	perror("tm_load failed");
	// 	return 1;
	// }
    
    // Load in data
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

    uint8_t *x_train = x_data;
    int32_t *y_train = y_data;
    uint8_t *x_test = x_data + 60000 * cols;
    int32_t *y_test = y_data + 60000;

    // Train models
    train_models(tm, NULL, x_train, y_train, 1000);

    // Evaluate models
    evaluate_models(tm, NULL, NULL, x_test, y_test, 1000);

	// Clean up
    tm_free(tm);
    // stm_free(stm);
    free(x_data);
    free(y_data);
    
    return 0;
}
