#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tm_output_activation.h"


// --- Basic output_activation functions ---

void oa_class_idx(const struct TsetlinMachine *tm, const int32_t *clause_votes, void *y_pred) {
    if (tm->y_size != 1) {
        fprintf(stderr, "y_eq_class_idx expects y_size == 1");
        exit(1);
    }
    uint32_t *label_pred = (uint32_t *)y_pred;

    // class index compare
    uint32_t best_class = 0;
    int32_t max_class_score = clause_votes[0];
    for (uint32_t class_id = 1; class_id < tm->num_classes; class_id++) {
        if (max_class_score < clause_votes[class_id]) {
            max_class_score = clause_votes[class_id];
            best_class = class_id;
        }
    }

    *label_pred = best_class;
}

void oa_bin_vector(const struct TsetlinMachine *tm, const int32_t *clause_votes, void *y_pred) {
    if(tm->y_size != tm->num_classes) {
        fprintf(stderr, "y_eq_bin_vector expects y_size == tm->num_classes");
        exit(1);
    }
    uint8_t *y_bin_vec = (uint8_t *)y_pred;

    for (uint32_t class_id = 0; class_id < tm->num_classes; class_id++) {
        // binary threshold (k=mid_state)
        y_bin_vec[class_id] = (clause_votes[class_id] >= tm->mid_state);
    }
}


// --- Basic output_activation_pseudograd functions ---
void oa_class_idx_pseudograd(const struct TsetlinMachine *tm, const void *y, const int32_t *clause_votes, int8_t *grad) {
    const uint32_t *label = (const uint32_t *)y;

    for (uint32_t class_id = 0; class_id < tm->num_classes; class_id++) {
        int32_t votes_clipped;
        votes_clipped = (clause_votes[class_id] > (int32_t)tm->threshold) ? (int32_t)tm->threshold : clause_votes[class_id];
        votes_clipped = (clause_votes[class_id] < -(int32_t)tm->threshold) ? -(int32_t)tm->threshold : clause_votes[class_id];
        
        float update_probability = ((float)votes_clipped + (float)tm->threshold) / (float)(2 * tm->threshold);

        grad[class_id] = (1.0 * rand()/RAND_MAX >= update_probability) ? -1 : 0;
    }
    grad[*label] = -grad[*label];
}
