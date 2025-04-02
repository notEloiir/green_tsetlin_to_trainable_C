#include <stdio.h>

#include "TsetlinMachine.h"


int main() {
    const char *filename = "mnist_tm.bin";
    struct TsetlinMachine *tm = load_tsetlin_machine(filename);

    if (tm) {
        printf("Threshold: %d\n", tm->threshold);
        printf("Features: %d\n", tm->n_literals);
        printf("Clauses: %d\n", tm->n_clauses);
        printf("Classes: %d\n", tm->n_classes);
        printf("Max state: %d\n", tm->max_state);
        printf("Min state: %d\n", tm->min_state);
        printf("Boost: %d\n", tm->boost_true_positive_feedback);

        free_tsetlin_machine(tm);
    }
    return 0;
}
