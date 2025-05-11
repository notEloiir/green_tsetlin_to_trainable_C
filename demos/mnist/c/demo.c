#include <stdio.h>
#include <stdlib.h>

#include "tsetlin_machine.h"
#include "sparse_tsetlin_machine.h"


// Loading data borrowed from https://github.com/ooki/green_tsetlin/blob/master/generator_tests/mnist_test.c
void load_mnist_data(uint8_t *x_data, int32_t *y_data) {
	int rows = 70000;
	int cols = 784;
	
	FILE *x_file = fopen("data/demos/mnist/mnist_x_70000_784.test_bin", "rb");
    if (x_file == NULL) {
        fprintf(stderr, "Failed to open x file\n");
        exit(1);
    }

    size_t x_read = fread(x_data, sizeof(uint8_t), rows * cols, x_file);
    if (x_read != (size_t) rows * cols) {
        fprintf(stderr, "Failed to read all data from x file\n");
        fclose(x_file);
        free(x_data);
        exit(1);
    }

    fclose(x_file);

    // Read y values
    FILE *y_file = fopen("data/demos/mnist/mnist_y_70000_784.test_bin", "rb");
    if (y_file == NULL) {
        fprintf(stderr, "Failed to open y file\n");
        free(x_data);
        exit(1);
    }

    size_t y_read = fread(y_data, sizeof(int32_t), rows, y_file);
    if (y_read != (size_t) rows) {
        fprintf(stderr, "Failed to read all data from y file\n");
        fclose(y_file);
        free(x_data);
        exit(1);
    }
    fclose(y_file);

    int h = 0;
    for(int col = 0; col < cols; col++)
    {
        //h += ((int)(x_data[k] * (k+1))) % 113;
        h += (int)x_data[col];
    }
    printf("hash: %d\n", h);
    return;
}


int main() {
    srand(42);

    const char *file_path = "data/models/mnist_tm.bin";
    struct TsetlinMachine *tm = tm_load(file_path, 1, sizeof(int32_t));
    struct SparseTsetlinMachine *stm = stm_load_dense(file_path, 1, sizeof(int32_t));
    if (tm == NULL || stm == NULL) {
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

    // Evaluate the loaded Tsetlin Machines
    rows = 5000;  // Evaluate on first 5000 rows
    printf("Evaluating Tsetlin Machine model\n");
    tm_evaluate(tm, x_data, y_data, rows);
    printf("Evaluating Sparse Tsetlin Machine model\n");
    stm_evaluate(stm, x_data, y_data, rows);

	// Clean up
    tm_free(tm);
    stm_free(stm);
    free(x_data);
    free(y_data);
    
    return 0;
}
