#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "MultiClassTsetlinMachine.h"

struct MultiClassTsetlinMachine* read_from_bin(const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Error opening file");
        return NULL;
    }
    
    int threshold;
    int n_literals, n_clauses, n_classes;
    int max_state, min_state;
    int boost_true_positive_feedback;

    // Read metadata
    fread(&threshold, sizeof(int), 1, file);
    fread(&n_literals, sizeof(int), 1, file);
    fread(&n_clauses, sizeof(int), 1, file);
    fread(&n_classes, sizeof(int), 1, file);
    fread(&max_state, sizeof(int), 1, file);
    fread(&min_state, sizeof(int), 1, file);
    fread(&boost_true_positive_feedback, sizeof(int), 1, file);
    
    struct MultiClassTsetlinMachine *tm = CreateMultiClassTsetlinMachine(
        n_classes, threshold, n_literals, n_clauses, max_state, min_state, boost_true_positive_feedback, 1, 0);
    if (!tm) {
        perror("CreateMultiClassTsetlinMachine failed");
        fclose(file);
        return NULL;
    }
    
    // TODO: figure out how to do this \/ \/ \/

    // Allocate and read weights
    size_t weights_size = n_clauses * n_classes * sizeof(int16_t);
    // fread(tm->weights, weights_size, 1, file);

    // Allocate and read clauses
    size_t clauses_size = n_clauses * n_literals * 2 * sizeof(int8_t);
    // fread(tm->clauses, clauses_size, 1, file);

    fclose(file);
    return tm;
}


int main() {
    const char *filename = "mnist_tm.bin";
    struct MultiClassTsetlinMachine *tm = read_from_bin(filename);

    if (tm) {
        printf("Threshold: %d\n", tm->tsetlin_machines[0]->threshold);
        printf("Features: %d\n", tm->tsetlin_machines[0]->n_literals);
        printf("Clauses: %d\n", tm->tsetlin_machines[0]->n_clauses);
        printf("Classes: %d\n", tm->n_classes);
        printf("Max state: %d\n", tm->tsetlin_machines[0]->max_state);
        printf("Min state: %d\n", tm->tsetlin_machines[0]->min_state);
        printf("Boost: %d\n", tm->tsetlin_machines[0]->boost_true_positive_feedback);

        free_mc_tsetlin_machine(tm);
    }
    return 0;
}
