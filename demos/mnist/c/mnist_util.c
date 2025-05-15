#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "mnist_util.h"


// Loading data borrowed from https://github.com/ooki/green_tsetlin/blob/master/generator_tests/mnist_test.c
void load_mnist_data(uint8_t *x_data, int32_t *y_data) {
	int rows = 70000;
	int cols = 784;
	
	FILE *x_file = fopen("data/demos/mnist/mnist_x_70000_784.bin", "rb");
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
    FILE *y_file = fopen("data/demos/mnist/mnist_y_70000_784.bin", "rb");
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


void evaluate_models(
    struct TsetlinMachine *tm,
    struct SparseTsetlinMachine *stm,
    struct StatelessTsetlinMachine *sltm,
    uint8_t *x_data,
    int32_t *y_data,
    uint32_t rows
) {
    // Evaluate the loaded Tsetlin Machines
    clock_t start_clock, end_clock;

    if (tm != NULL) {
        printf("Evaluating Tsetlin Machine model\n");
        start_clock = clock();
        tm_evaluate(tm, x_data, y_data, rows);
        end_clock = clock();
        printf("Tsetlin Machine time: %f[s]\n", ((double) (end_clock - start_clock)) / CLOCKS_PER_SEC);
    }

    if (stm != NULL) {
        printf("Evaluating Sparse Tsetlin Machine model\n");
        start_clock = clock();
        stm_evaluate(stm, x_data, y_data, rows);
        end_clock = clock();
        printf("Sparse Tsetlin Machine time: %f[s]\n", ((double) (end_clock - start_clock)) / CLOCKS_PER_SEC);
    }

    if (sltm != NULL) {
        printf("Evaluating Stateless (Sparse) Tsetlin Machine model\n");
        start_clock = clock();
        sltm_evaluate(sltm, x_data, y_data, rows);
        end_clock = clock();
        printf("Stateless (Sparse) Tsetlin Machine time: %f[s]\n", ((double) (end_clock - start_clock)) / CLOCKS_PER_SEC);
    }
}

void train_models(
    struct TsetlinMachine *tm,
    struct SparseTsetlinMachine *stm,
    uint8_t *x_data,
    int32_t *y_data,
    uint32_t rows
) {
    // Train the Tsetlin Machines
    clock_t start_clock, end_clock;

    if (tm != NULL) {
        printf("Training Tsetlin Machine model\n");
        start_clock = clock();
        tm_train(tm, x_data, y_data, rows, 10, 2);
        end_clock = clock();
        printf("Tsetlin Machine time: %f[s]\n", ((double) (end_clock - start_clock)) / CLOCKS_PER_SEC);
    }

    if (stm != NULL) {
        printf("Training Sparse Tsetlin Machine model\n");
        start_clock = clock();
        stm_train(stm, x_data, y_data, rows, 10, 2);
        end_clock = clock();
        printf("Sparse Tsetlin Machine time: %f[s]\n", ((double) (end_clock - start_clock)) / CLOCKS_PER_SEC);
    }
}
