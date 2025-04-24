#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tm_output_activation.h"
#include "tsetlin_machine.h"
#include "utility.h"


// --- Basic output_activation functions ---

void oa_class_idx(const struct TsetlinMachine *tm, void *y_pred) {
    if (tm->y_size != 1) {
        fprintf(stderr, "y_eq_class_idx expects y_size == 1");
        exit(1);
    }
    uint32_t *label_pred = (uint32_t *)y_pred;

    // class index compare
    uint32_t best_class = 0;
    int32_t max_class_score = tm->votes[0];
    for (uint32_t class_id = 1; class_id < tm->num_classes; class_id++) {
        if (max_class_score < tm->votes[class_id]) {
            max_class_score = tm->votes[class_id];
            best_class = class_id;
        }
    }

    *label_pred = best_class;
}

void oa_bin_vector(const struct TsetlinMachine *tm, void *y_pred) {
    if(tm->y_size != tm->num_classes) {
        fprintf(stderr, "y_eq_bin_vector expects y_size == tm->num_classes");
        exit(1);
    }
    uint8_t *y_bin_vec = (uint8_t *)y_pred;

    for (uint32_t class_id = 0; class_id < tm->num_classes; class_id++) {
        // binary threshold (k=mid_state)
        y_bin_vec[class_id] = (tm->votes[class_id] > tm->mid_state);
    }
}


// --- Basic output_activation_pseudograd functions ---

void feedback_class_idx(const struct TsetlinMachine *tm, const void *y, uint32_t clause_id) {
    const uint32_t *label = (const uint32_t *)y;

    // Correct label gets feedback type 1a or 1b, incorrect maybe get type 2 (depending on clause output)
    for (uint32_t class_id = 0; class_id < tm->num_classes; class_id++) {
        int32_t votes_clipped = clip(tm->votes[class_id], (int32_t)tm->threshold);
        float update_probability = ((float)votes_clipped + (float)tm->threshold) / (float)(2 * tm->threshold);
        int8_t *clause_feedback = tm->feedback + ((clause_id * tm->num_classes + class_id) * 3);
        uint8_t feedback_strength = (1.0 * rand()/RAND_MAX >= update_probability);
        
        if (class_id == *label) {
            // Correct vote
            if (tm->clause_output[clause_id] == 1) {
                clause_feedback[0] += feedback_strength;
            }
            else {
                clause_feedback[1] += feedback_strength;
            }
        }
        else if (tm->clause_output[clause_id] == 1) {
            clause_feedback[2] += feedback_strength;
        }
    }
}

void feedback_bin_vector(const struct TsetlinMachine *tm, const void *y, uint32_t clause_id) {
    uint8_t *y_bin_vec = (uint8_t *)y;

    // Correct votes gets feedback type 1a or 1b, incorrect maybe get type 2 (depending on clause output)
    for (uint32_t class_id = 0; class_id < tm->num_classes; class_id++) {
        int32_t votes_clipped = clip(tm->votes[class_id], (int32_t)tm->threshold);
        float update_probability = ((float)votes_clipped + (float)tm->threshold) / (float)(2 * tm->threshold);
        int8_t *clause_feedback = tm->feedback + ((clause_id * tm->num_classes + class_id) * 3);
        uint8_t feedback_strength = (1.0 * rand()/RAND_MAX >= update_probability);

        if ((tm->votes[class_id] > tm->mid_state) == y_bin_vec[class_id]) {
            // Correct vote
            if (tm->clause_output[clause_id] == 1) {
                clause_feedback[0] += feedback_strength;
            }
            else {
                clause_feedback[1] += feedback_strength;
            }
        }
        else if (tm->clause_output[clause_id] == 1) {
            clause_feedback[2] += feedback_strength;
        }
    }
}
