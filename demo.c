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
    int n_features;
    int n_clauses;
    int n_classes;
    int n_states;
    int boost_true_positive_feedback;

    // Read metadata
    fread(&threshold, sizeof(int), 1, file);
    fread(&n_features, sizeof(int), 1, file);
    fread(&n_clauses, sizeof(int), 1, file);
    fread(&n_classes, sizeof(int), 1, file);
    fread(&n_states, sizeof(int), 1, file);
    fread(&boost_true_positive_feedback, sizeof(int), 1, file);
    
    struct MultiClassTsetlinMachine *tm = CreateMultiClassTsetlinMachine(n_classes, threshold, 
    	n_features, n_clauses, n_states, boost_true_positive_feedback, 1, 0);
    if (!tm) {
        perror("CreateMultiClassTsetlinMachine failed");
        fclose(file);
        return NULL;
    }
    
    // TODO: figure out how to do this \/ \/ \/

    // Allocate and read weights
    size_t weights_size = tm->n_clauses * tm->n_classes * sizeof(int16_t);
    // fread(tm->weights, weights_size, 1, file);

    // Allocate and read clauses
    size_t clauses_size = tm->n_clauses * tm->n_features * 2 * sizeof(int8_t);
    // fread(tm->clauses, clauses_size, 1, file);

    fclose(file);
    return tm;
}


int main() {
    const char *filename = "mnist_tm.bin";
    struct MultiClassTsetlinMachine *tm = read_from_bin(filename);

    if (tm) {
        printf("Threshold: %d\n", tm->threshold);
        printf("Features: %d\n", tm->n_features);
        printf("Clauses: %d\n", tm->n_clauses);
        printf("Classes: %d\n", tm->n_classes);
        printf("States: %d\n", tm->n_states);
        printf("Boost: %d\n", tm->boost_true_positive_feedback);

        free_mc_tsetlin_machine(tm);
    }
    return 0;
}
