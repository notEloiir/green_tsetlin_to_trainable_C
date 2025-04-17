#include <stdio.h>
#include <stdlib.h>

#include "TsetlinMachine.h"


// Loading data borrowed from https://github.com/ooki/green_tsetlin/blob/master/generator_tests/mnist_test.c
void load_mnist_data(uint8_t *x_data, uint32_t *y_data) {
	int rows = 70000;
	int cols = 784;
	
	FILE *x_file = fopen("data/demos/mnist/mnist_x_70000_784.test_bin", "rb");
    if (x_file == NULL) {
        printf("Failed to open x file\n");
        exit(1);
    }

    size_t x_read = fread(x_data, sizeof(uint8_t), rows * cols, x_file);
    if (x_read != (size_t) rows * cols) {
        printf("Failed to read all data from x file\n");
        fclose(x_file);
        free(x_data);
        exit(1);
    }

    fclose(x_file);

    // Read y values
    FILE *y_file = fopen("data/demos/mnist/mnist_y_70000_784.test_bin", "rb");
    if (y_file == NULL) {
        printf("Failed to open y file\n");
        free(x_data);
        exit(1);
    }

    size_t y_read = fread(y_data, sizeof(uint32_t), rows, y_file);
    if (y_read != (size_t) rows) {
        printf("Failed to read all data from y file\n");
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


void eval_model(struct TsetlinMachine *tm, uint8_t *X, uint32_t *y, int rows, int cols) {
	int correct = 0;
    int total = 0;
    int *result = malloc(tm->num_classes * sizeof(int));
    if (result == NULL) {
        printf("Failed to allocate memory for result\n");
        exit(1);
    }
    
    for(int row = 0; row < rows; ++row)
    {
		if (row % 100 == 0 && row) {
			printf("%d out of %d done\n", row, rows);
		}
		
		uint8_t* example = &X[row * cols];
		
		tm_score(tm, example, result);
		
		uint32_t best_class = 0;
		int max_class_score = result[0];
		for (uint32_t class_id = 1; class_id < (uint32_t)tm->num_classes; class_id++) {
			if (max_class_score < result[class_id]) {
				max_class_score = result[class_id];
				best_class = class_id;
			}
		}
		
        if(best_class == y[row])
            correct += 1;
        
        total += 1;
    }
    printf("correct: %d, total: %d, ratio: %.2f \n", correct, total, (float) correct / total);
}


int main() {
    const char *file_path = "data/models/mnist_tm.bin";
    struct TsetlinMachine *tm = load_tsetlin_machine(file_path);
    if (tm == NULL) {
		perror("load_tsetlin_machine failed");
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
    
    // Load in test data
	int rows = 70000;
	int cols = 784;
    uint8_t *x_data = malloc(rows * cols * sizeof(uint8_t));
    uint32_t *y_data = malloc(rows * sizeof(uint32_t));
    if (x_data == NULL || y_data == NULL) {
        printf("Failed to allocate memory for x_data or y_data\n");
        return 1;
    }
    printf("Loading MNIST data\n");
    load_mnist_data(x_data, y_data);
    
    // Evaluate the loaded Tsetlin Machine
    printf("Evaluating model\n");
    rows = 1000;  // Evaluate on 1000 first rows
    eval_model(tm, x_data, y_data, rows, cols);

	// Clean up
    free_tsetlin_machine(tm);
    free(x_data);
    free(y_data);
    
    return 0;
}
