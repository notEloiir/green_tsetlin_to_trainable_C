#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#include "tsetlin_machine.h"
#include "utility.h"


// --- Basic y_eq function ---

uint8_t tm_y_eq_generic(const struct TsetlinMachine *tm, const void *y, const void *y_pred) {
    return 0 == memcmp(y, y_pred, tm->y_size * tm->y_element_size);
}


// --- Tsetlin Machine ---

void tm_initialize(struct TsetlinMachine *tm);

// Allocate memory, fill in fields, calls tm_initialize
struct TsetlinMachine *tm_create(
    uint32_t num_classes, uint32_t threshold, uint32_t num_literals, uint32_t num_clauses,
    int8_t max_state, int8_t min_state, uint8_t boost_true_positive_feedback,
    uint32_t y_size, uint32_t y_element_size, float s
) {
    struct TsetlinMachine *tm = (struct TsetlinMachine *)malloc(sizeof(struct TsetlinMachine));
    if(tm == NULL) {
        perror("Memory allocation failed");
        return NULL;
    }
    
    tm->num_classes = num_classes;
    tm->threshold = threshold;
    tm->num_literals = num_literals;
    tm->num_clauses = num_clauses;
    tm->max_state = max_state;
    tm->min_state = min_state;
    tm->boost_true_positive_feedback = boost_true_positive_feedback;
    tm->s = s;
    
    tm->y_size = y_size;
    tm->y_element_size = y_element_size;
    tm->y_eq = tm_y_eq_generic;
    tm->output_activation = tm_oa_class_idx;
    tm->calculate_feedback = tm_feedback_class_idx;
    
    tm->ta_state = (int8_t *)malloc(num_clauses * num_literals * 2 * sizeof(int8_t));  // shape: flat (num_clauses, num_literals, 2)
    if (tm->ta_state == NULL) {
        perror("Memory allocation failed");
        tm_free(tm);
        return NULL;
    }
    
    tm->weights = (int16_t *)malloc(num_clauses * num_classes * sizeof(int16_t));  // shape: flat (num_clauses, num_classes)
    if (tm->weights == NULL) {
        perror("Memory allocation failed");
        tm_free(tm);
        return NULL;
    }
    
    tm->clause_output = (uint8_t *)malloc(num_clauses * sizeof(uint8_t));  // shape: (num_clauses)
    if (tm->clause_output == NULL) {
        perror("Memory allocation failed");
        tm_free(tm);
        return NULL;
    }

    tm->votes = (int32_t *)malloc(num_classes * sizeof(int32_t));  // shape: (num_classes)
    if (tm->votes == NULL) {
        perror("Memory allocation failed");
        tm_free(tm);
        return NULL;
    }

    /* Set up the Tsetlin Machine structure */

    tm_initialize(tm);
    
    return tm;
}


// Load Tsetlin Machine from a bin file
struct TsetlinMachine *tm_load(
    const char *filename, uint32_t y_size, uint32_t y_element_size
) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Error opening file");
        return NULL;
    }
    
    uint32_t threshold, num_literals, num_clauses, num_classes;
    int8_t max_state, min_state;
    uint8_t boost_true_positive_feedback;
    double s_double;

    size_t threshold_read, num_literals_read, num_clauses_read, num_classes_read;
    size_t max_state_read, min_state_read, boost_true_positive_feedback_read, s_double_read;

    // Read metadata
    threshold_read = fread(&threshold, sizeof(uint32_t), 1, file);
    num_literals_read = fread(&num_literals, sizeof(uint32_t), 1, file);
    num_clauses_read = fread(&num_clauses, sizeof(uint32_t), 1, file);
    num_classes_read = fread(&num_classes, sizeof(uint32_t), 1, file);
    max_state_read = fread(&max_state, sizeof(int8_t), 1, file);
    min_state_read = fread(&min_state, sizeof(int8_t), 1, file);
    boost_true_positive_feedback_read = fread(&boost_true_positive_feedback, sizeof(uint8_t), 1, file);
    s_double_read = fread(&s_double, sizeof(double), 1, file);

    if (threshold_read != 1 || num_literals_read != 1 || num_clauses_read != 1 || num_classes_read != 1 ||
            max_state_read != 1 || min_state_read != 1 ||
            boost_true_positive_feedback_read != 1 || s_double_read != 1) {
        fprintf(stderr, "Failed to read all metadata from bin\n");
        fclose(file);
        return NULL;
    }
    
    struct TsetlinMachine *tm = tm_create(
        num_classes, threshold, num_literals, num_clauses,
        max_state, min_state, boost_true_positive_feedback,
        y_size, y_element_size, (float)s_double
    );
    if (!tm) {
        fprintf(stderr, "tm_create failed\n");
        fclose(file);
        return NULL;
    }

    // Allocate and read weights
    size_t weights_read = fread(tm->weights, sizeof(int16_t), num_clauses * num_classes, file);
    if (weights_read != num_clauses * num_classes) {
        fprintf(stderr, "Failed to read all weights from bin\n");
        tm_free(tm);
        fclose(file);
        return NULL;
    }

    // Allocate and read clauses
    size_t states_read = fread(tm->ta_state, sizeof(int8_t), num_clauses * num_literals * 2, file);
    if (states_read != num_clauses * num_literals * 2) {
        fprintf(stderr, "Failed to read all states from bin\n");
        tm_free(tm);
        fclose(file);
        return NULL;
    }

    fclose(file);
    return tm;
}


void tm_save(const struct TsetlinMachine *tm, const char *filename) {
    FILE *file = fopen(filename, "wb");
    if (!file) {
        perror("Error opening file for writing");
        return;
    }

    size_t written;

    written = fwrite(&tm->threshold, sizeof(uint32_t), 1, file);
    if (written != 1) {
        fprintf(stderr, "Failed to write threshold\n");
        goto save_error;
    }
    written = fwrite(&tm->num_literals, sizeof(uint32_t), 1, file);
    if (written != 1) {
        fprintf(stderr, "Failed to write num_literals\n");
        goto save_error;
    }
    written = fwrite(&tm->num_clauses, sizeof(uint32_t), 1, file);
    if (written != 1) {
        fprintf(stderr, "Failed to write num_clauses\n");
        goto save_error;
    }
    written = fwrite(&tm->num_classes, sizeof(uint32_t), 1, file);
    if (written != 1) {
        fprintf(stderr, "Failed to write num_classes\n");
        goto save_error;
    }
    written = fwrite(&tm->max_state, sizeof(int8_t), 1, file);
    if (written != 1) {
        fprintf(stderr, "Failed to write max_state\n");
        goto save_error;
    }
    written = fwrite(&tm->min_state, sizeof(int8_t), 1, file);
    if (written != 1) {
        fprintf(stderr, "Failed to write min_state\n");
        goto save_error;
    }
    written = fwrite(&tm->boost_true_positive_feedback, sizeof(uint8_t), 1, file);
    if (written != 1) {
        fprintf(stderr, "Failed to write boost_true_positive_feedback\n");
        goto save_error;
    }
    written = fwrite(&tm->s, sizeof(double), 1, file);
    if (written != 1) {
        fprintf(stderr, "Failed to write s parameter\n");
        goto save_error;
    }

    size_t n_weights = (size_t)tm->num_clauses * tm->num_classes;
    written = fwrite(tm->weights, sizeof(int16_t), n_weights, file);
    if (written != n_weights) {
        fprintf(stderr, "Failed to write weights array (%zu of %zu)\n",
                written, n_weights);
        goto save_error;
    }

    size_t n_states = (size_t)tm->num_clauses * tm->num_literals * 2;
    written = fwrite(tm->ta_state, sizeof(int8_t), n_states, file);
    if (written != n_states) {
        fprintf(stderr, "Failed to write ta_state array (%zu of %zu)\n",
                written, n_states);
        goto save_error;
    }

    fclose(file);
    return;

save_error:
    fclose(file);
    fprintf(stderr, "tm_save aborted, file %s may be incomplete\n", filename);
}


// Free all allocated memory
void tm_free(struct TsetlinMachine *tm) {
    if (tm != NULL){
        if (tm->ta_state != NULL) {
            free(tm->ta_state);
        }
        
        if (tm->weights != NULL) {
            free(tm->weights);
        }
        
        if (tm->clause_output != NULL) {
            free(tm->clause_output);
        }
        
        if (tm->votes != NULL) {
            free(tm->votes);
        }
        
        free(tm);
    }
    
    return;
}


// Initialize values
void tm_initialize(struct TsetlinMachine *tm) {
    tm->mid_state = (tm->max_state + tm->min_state) / 2;
    tm->s_inv = 1.0f / tm->s;
    tm->s_min1_inv = (tm->s - 1.0f) / tm->s;

    for (uint32_t clause_id = 0; clause_id < tm->num_clauses; clause_id++) {				
        for (uint32_t literal_id = 0; literal_id < tm->num_literals; literal_id++) {
            if (1.0 * rand()/RAND_MAX <= 0.5) {
                // positive literal
                tm->ta_state[(((clause_id * tm->num_literals) + literal_id) * 2) + 0] = tm->mid_state - 1;
                // negative literal
                tm->ta_state[(((clause_id * tm->num_literals) + literal_id) * 2) + 1] = tm->mid_state;
            } else {
                tm->ta_state[(((clause_id * tm->num_literals) + literal_id) * 2) + 0] = tm->mid_state;
                tm->ta_state[(((clause_id * tm->num_literals) + literal_id) * 2) + 1] = tm->mid_state - 1;
            }
        }
    }
    
    // Init weights randomly to -1 or 1
    for (uint32_t clause_id = 0; clause_id < tm->num_clauses; clause_id++) {
        for (uint32_t class_id = 0; class_id < tm->num_classes; class_id++) {
            tm->weights[(clause_id * tm->num_classes) + class_id] = 1 - 2*(1.0 * rand()/RAND_MAX <= 0.5);
        }
    }
}

// Translates automaton state to action - 0 or 1
static inline uint8_t action(int8_t state, int8_t mid_state) {
    return state >= mid_state;
}

// Calculate the output of each clause using the actions of each Tsetlin Automaton
// Output is stored an internal output array clause_output
static inline void calculate_clause_output(struct TsetlinMachine *tm, const uint8_t *X, uint8_t skip_empty) {
    // For each clause, check if it is "active" - all necessary literals have the right value
    for (uint32_t clause_id = 0; clause_id < tm->num_clauses; clause_id++) {
        tm->clause_output[clause_id] = 1;
        uint8_t empty_clause = 1;

        for (uint32_t literal_id = 0; literal_id < tm->num_literals; literal_id++) {
            uint8_t action_include = action(tm->ta_state[(((clause_id * tm->num_literals) + literal_id) * 2) + 0], tm->mid_state);
            uint8_t action_include_negated = action(tm->ta_state[(((clause_id * tm->num_literals) + literal_id) * 2) + 1], tm->mid_state);
            
            empty_clause = (empty_clause && !(action_include || action_include_negated));

            if ((action_include == 1 && X[literal_id] == 0) || (action_include_negated == 1 && X[literal_id] == 1)) {
                tm->clause_output[clause_id] = 0;
                break;
            }
        }

        if (empty_clause && skip_empty) {
            tm->clause_output[clause_id] = 0;
        }
    }
}


// Sum up the votes of each clause for each class
static inline void sum_votes(struct TsetlinMachine *tm) {
    memset(tm->votes, 0, tm->num_classes*sizeof(int32_t));
    
    for (uint32_t clause_id = 0; clause_id < tm->num_clauses; clause_id++) {
        if (tm->clause_output[clause_id] == 0) {
            continue;
        }
        
        for (uint32_t class_id = 0; class_id < tm->num_classes; class_id++) {
            tm->votes[class_id] += tm->weights[(clause_id * tm->num_classes) + class_id];
        }
    }
    
    for (uint32_t class_id = 0; class_id < tm->num_classes; class_id++) {
        tm->votes[class_id] = clip(tm->votes[class_id], (int32_t)tm->threshold);
    }
}


// Type I Feedback
// Clause at clause_id voted correctly for class at class_id

// Type a - Clause is active for literals X (clause_output == 1)
static inline void type_1a_feedback(struct TsetlinMachine *tm, const uint8_t *X, uint32_t clause_id, uint32_t class_id) {
    // float s_inv = 1.0f / tm->s;
    // float s_min1_inv = (tm->s - 1.0f) / tm->s;

    uint8_t feedback_strength = 1;

    if (tm->weights[clause_id * tm->num_classes + class_id] >= 0) {
        tm->weights[clause_id * tm->num_classes + class_id] += min(feedback_strength, SHRT_MAX - tm->weights[clause_id * tm->num_classes + class_id]);
    }
    else {
        tm->weights[clause_id * tm->num_classes + class_id] -= min(feedback_strength, -(SHRT_MIN - tm->weights[clause_id * tm->num_classes + class_id]));
    }
    
    for (uint32_t literal_id = 0; literal_id < tm->num_literals; literal_id++) {
        if (X[literal_id] == 1) {
            // True positive
            tm->ta_state[(((clause_id * tm->num_literals) + literal_id) * 2)] +=
				min(tm->max_state - tm->ta_state[(((clause_id * tm->num_literals) + literal_id) * 2)], feedback_strength) * (
                (tm->boost_true_positive_feedback == 1 || 1.0*rand()/RAND_MAX <= tm->s_min1_inv));

            // False negative
            tm->ta_state[(((clause_id * tm->num_literals) + literal_id) * 2) + 1] -=
				min(-(tm->min_state - tm->ta_state[(((clause_id * tm->num_literals) + literal_id) * 2) + 1]), feedback_strength) * (
                (1.0*rand()/RAND_MAX <= tm->s_inv));

        } else {
            // True negative
            tm->ta_state[(((clause_id * tm->num_literals) + literal_id) * 2) + 1] +=
				min(tm->max_state - tm->ta_state[(((clause_id * tm->num_literals) + literal_id) * 2) + 1], feedback_strength) * (
                (1.0*rand()/RAND_MAX <= tm->s_min1_inv));
            
            // False positive
            tm->ta_state[(((clause_id * tm->num_literals) + literal_id) * 2)] -=
				min(-(tm->min_state - tm->ta_state[(((clause_id * tm->num_literals) + literal_id) * 2)]), feedback_strength) * (
                (tm->ta_state[(((clause_id * tm->num_literals) + literal_id) * 2)] > tm->min_state) && 
                (1.0*rand()/RAND_MAX <= tm->s_inv));
        }
    }
}


// Type b - Clause is inactive for literals X (clause_output == 0)
static inline void type_1b_feedback(struct TsetlinMachine *tm, uint32_t clause_id, uint32_t class_id) {
    // float s_inv = 1.0f / tm->s;

    uint8_t feedback_strength = 1;

    for (uint32_t literal_id = 0; literal_id < tm->num_literals; literal_id++) {
        tm->ta_state[(((clause_id * tm->num_literals) + literal_id) * 2)] -=
			min(-(tm->min_state - tm->ta_state[(((clause_id * tm->num_literals) + literal_id) * 2)]), feedback_strength) * (
            (1.0*rand()/RAND_MAX <= tm->s_inv));

        tm->ta_state[(((clause_id * tm->num_literals) + literal_id) * 2) + 1] -=
			min(-(tm->min_state - tm->ta_state[(((clause_id * tm->num_literals) + literal_id) * 2) + 1]), feedback_strength) * (
            (1.0*rand()/RAND_MAX <= tm->s_inv));
    }
}


// Type II Feedback
// Clause at clause_id voted incorrectly for class at class_id
// && Clause is active for literals X (clause_output == 1)

static inline void type_2_feedback(struct TsetlinMachine *tm, const uint8_t *X, uint32_t clause_id, uint32_t class_id) {
    uint8_t feedback_strength = 1;

    tm->weights[clause_id * tm->num_classes + class_id] +=
        tm->weights[clause_id * tm->num_classes + class_id] >= 0 ? -feedback_strength : feedback_strength;

    for (uint32_t literal_id = 0; literal_id < tm->num_literals; literal_id++) {
        tm->ta_state[(((clause_id * tm->num_literals) + literal_id) * 2)] +=
			min(tm->max_state - tm->ta_state[(((clause_id * tm->num_literals) + literal_id) * 2)], feedback_strength) * (
            0 == action(tm->ta_state[(((clause_id * tm->num_literals) + literal_id) * 2)], tm->mid_state) &&
            0 == X[literal_id]);

        tm->ta_state[(((clause_id * tm->num_literals) + literal_id) * 2) + 1] +=
			min(tm->max_state - tm->ta_state[(((clause_id * tm->num_literals) + literal_id) * 2) + 1], feedback_strength) * (
            0 == action(tm->ta_state[(((clause_id * tm->num_literals) + literal_id) * 2) + 1], tm->mid_state) &&
            1 == X[literal_id]);
    }
}


void tm_train(struct TsetlinMachine *tm, const uint8_t *X, const void *y, uint32_t rows, uint32_t epochs) {
    for (uint32_t epoch = 0; epoch < epochs; epoch++) {
		for (uint32_t row = 0; row < rows; row++) {
			const uint8_t *X_row = X + (row * tm->num_literals);
			void *y_row = (void *)((uint8_t *)y + (row * tm->y_size * tm->y_element_size));

			calculate_clause_output(tm, X_row, 0);

			sum_votes(tm);

			// Iterate over all clauses, not only active ones (1b)
			// Calculate pseudo gradient - feedback to clause-class vote weight
			for (uint32_t clause_id = 0; clause_id < tm->num_clauses; clause_id++) {
				tm->calculate_feedback(tm, X_row, y_row, clause_id);
			}
        }
    }
}


// Inference
// y_pred should be allocated like: void *y_pred = malloc(rows * tm->y_size * tm->y_element_size);
void tm_predict(struct TsetlinMachine *tm, const uint8_t *X, void *y_pred, uint32_t rows) {
    for (uint32_t row = 0; row < rows; row++) {
    	const uint8_t* X_row = X + (row * tm->num_literals);
        void *y_pred_row = (void *)(((uint8_t *)y_pred) + (row * tm->y_size * tm->y_element_size));

        // Calculate clause output
        calculate_clause_output(tm, X_row, 1);

        // Sum up clause votes for each class
        sum_votes(tm);

        // Pass through output activation function
        tm->output_activation(tm, y_pred_row);
    }
}


void tm_evaluate(struct TsetlinMachine *tm, const uint8_t *X, const void *y, uint32_t rows) {
    uint32_t correct = 0;
    uint32_t total = 0;
    void *y_pred = malloc(rows * tm->y_size * tm->y_element_size);
    if (y_pred == NULL) {
        perror("Memory allocation failed\n");
        exit(1);
    }

    tm_predict(tm, X, y_pred, rows);
    
    for(uint32_t row = 0; row < rows; ++row) {

        void* y_row = (void *)(((uint8_t *)y) + (row * tm->y_size * tm->y_element_size));
        void* y_pred_row = (void *)(((uint8_t *)y_pred) + (row * tm->y_size * tm->y_element_size));
        
        if (tm->y_eq(tm, y_row, y_pred_row)) {
            correct++;
        }
        total++;
    }
    printf("correct: %d, total: %d, ratio: %.2f \n", correct, total, (float) correct / total);
    free(y_pred);
}


// --- Basic output_activation functions ---

void tm_oa_class_idx(const struct TsetlinMachine *tm, const void *y_pred) {
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

void tm_oa_bin_vector(const struct TsetlinMachine *tm, const void *y_pred) {
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


void tm_set_output_activation(
    struct TsetlinMachine *tm,
    void (*output_activation)(const struct TsetlinMachine *tm, const void *y_pred)
) {
    tm->output_activation = output_activation;
}


// Internal component of feedback functions below
void tm_apply_feedback(struct TsetlinMachine *tm, uint32_t clause_id, uint32_t class_id, uint8_t is_class_positive, const uint8_t *X) {
	uint8_t is_vote_positive = tm->weights[(clause_id * tm->num_classes) + class_id] >= 0;
	if (is_vote_positive == is_class_positive) {
		if (tm->clause_output[clause_id] == 1) {
			type_1a_feedback(tm, X, clause_id, class_id);
		}
		else {
			type_1b_feedback(tm, clause_id, class_id);
		}
	}
	else if (tm->clause_output[clause_id] == 1) {
		type_2_feedback(tm, X, clause_id, class_id);
	}
}

// --- calculate_feedback ---
// Calculate clause-class feedback

void tm_feedback_class_idx(struct TsetlinMachine *tm, const uint8_t *X, const void *y, uint32_t clause_id) {
    // Correct label gets feedback type 1a or 1b, incorrect maybe get type 2 (depending on clause output)
    const uint32_t *label_ptr = (const uint32_t *)y;
    const uint32_t positive_class = *label_ptr;
    uint32_t negative_class = 0;

    int32_t votes_clipped_positive = clip(tm->votes[positive_class], (int32_t)tm->threshold);
    float update_probability_positive = ((float)tm->threshold - (float)votes_clipped_positive) / (float)(2 * tm->threshold);

    if (1.0 * rand()/RAND_MAX <= update_probability_positive) {
    	tm_apply_feedback(tm, clause_id, positive_class, 1, X);
    }

    int32_t sum_votes_clipped_negative = 0;
    for (uint32_t class_id = 0; class_id < tm->num_classes; class_id++) {
        if (class_id != positive_class) {
            sum_votes_clipped_negative += clip(tm->votes[class_id], (int32_t)tm->threshold) + (int32_t)tm->threshold;
        }
    }
    if (sum_votes_clipped_negative == 0) return;
    int32_t random_vote_negative = rand() % sum_votes_clipped_negative;
    int32_t accumulated_votes = 0;
    for (uint32_t class_id = 0; class_id < tm->num_classes; class_id++) {
        if (class_id != positive_class) {
            accumulated_votes += clip(tm->votes[class_id], (int32_t)tm->threshold) + (int32_t)tm->threshold;
            if (accumulated_votes >= random_vote_negative) {
                negative_class = class_id;
                break;
            }
        }
    }

    int32_t votes_clipped_negative = clip(tm->votes[negative_class], (int32_t)tm->threshold);
    float update_probability_negative = ((float)votes_clipped_negative + (float)tm->threshold) / (float)(2 * tm->threshold);

    if (1.0 * rand()/RAND_MAX <= update_probability_negative) {
    	tm_apply_feedback(tm, clause_id, negative_class, 0, X);
    }
}

void tm_feedback_bin_vector(struct TsetlinMachine *tm, const uint8_t *X, const void *y, uint32_t clause_id) {
    const uint8_t *label_arr = (const uint8_t *)y;
    uint32_t positive_class = 0;
    uint32_t negative_class = 0;

    int32_t sum_votes_clipped_positive = 0;
	for (uint32_t class_id = 0; class_id < tm->num_classes; class_id++) {
		if (label_arr[class_id]) {
			sum_votes_clipped_positive += clip(tm->votes[class_id], (int32_t)tm->threshold) + (int32_t)tm->threshold;
		}
	}
	if (sum_votes_clipped_positive == 0) goto negative_feedback;
	int32_t random_vote_positive = rand() % sum_votes_clipped_positive;
	int32_t accumulated_votes_positive = 0;
	for (uint32_t class_id = 0; class_id < tm->num_classes; class_id++) {
		if (label_arr[class_id]) {
			accumulated_votes_positive += clip(tm->votes[class_id], (int32_t)tm->threshold) + (int32_t)tm->threshold;
			if (accumulated_votes_positive >= random_vote_positive) {
				positive_class = class_id;
				break;
			}
		}
	}

	int32_t votes_clipped_positive = clip(tm->votes[negative_class], (int32_t)tm->threshold);
	float update_probability_positive = ((float)tm->threshold - (float)votes_clipped_positive) / (float)(2 * tm->threshold);

    if (1.0 * rand()/RAND_MAX <= update_probability_positive) {
    	tm_apply_feedback(tm, clause_id, positive_class, 1, X);
    }

negative_feedback:

    int32_t sum_votes_clipped_negative = 0;
	for (uint32_t class_id = 0; class_id < tm->num_classes; class_id++) {
		if (!label_arr[class_id]) {
			sum_votes_clipped_negative += clip(tm->votes[class_id], (int32_t)tm->threshold) + (int32_t)tm->threshold;
		}
	}
	if (sum_votes_clipped_negative == 0) return;
	int32_t random_vote_negative = rand() % sum_votes_clipped_negative;
	int32_t accumulated_votes_negative = 0;
	for (uint32_t class_id = 0; class_id < tm->num_classes; class_id++) {
		if (!label_arr[class_id]) {
			accumulated_votes_negative += clip(tm->votes[class_id], (int32_t)tm->threshold) + (int32_t)tm->threshold;
			if (accumulated_votes_negative >= random_vote_negative) {
				negative_class = class_id;
				break;
			}
		}
	}

	int32_t votes_clipped_negative = clip(tm->votes[negative_class], (int32_t)tm->threshold);
	float update_probability_negative = ((float)votes_clipped_negative + (float)tm->threshold) / (float)(2 * tm->threshold);

	if (1.0 * rand()/RAND_MAX <= update_probability_negative) {
		tm_apply_feedback(tm, clause_id, negative_class, 0, X);
	}
}


void tm_set_calculate_feedback(
    struct TsetlinMachine *tm,
    void (*calculate_feedback)(struct TsetlinMachine *tm, const uint8_t *X, const void *y, uint32_t clause_id)
) {
    tm->calculate_feedback = calculate_feedback;
}
