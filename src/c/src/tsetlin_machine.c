#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tsetlin_machine.h"
#include "tm_output_activation.h"
#include "utility.h"


// --- Basic y_eq function ---

uint8_t y_eq_generic(const struct TsetlinMachine *tm, const void *y, const void *y_pred) {
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
    tm->y_eq = y_eq_generic;
    tm->output_activation = oa_class_idx;
    tm->output_activation_pseudograd = oa_class_idx_pseudograd;
    
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
    
    tm->pseudograd = (int8_t *)malloc(num_clauses * num_classes * sizeof(int8_t));  // shape: (num_clauses, num_classes)
    if (tm->pseudograd == NULL) {
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
    const char *filename,
    uint32_t y_size, uint32_t y_element_size, float s
) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Error opening file");
        return NULL;
    }
    
    uint32_t threshold, num_literals, num_clauses, num_classes;
    int8_t max_state, min_state;
    uint8_t boost_true_positive_feedback;

    size_t threshold_read, num_literals_read, num_clauses_read, num_classes_read;
    size_t max_state_read, min_state_read, boost_true_positive_feedback_read;

    // Read metadata
    threshold_read = fread(&threshold, sizeof(uint32_t), 1, file);
    num_literals_read = fread(&num_literals, sizeof(uint32_t), 1, file);
    num_clauses_read = fread(&num_clauses, sizeof(uint32_t), 1, file);
    num_classes_read = fread(&num_classes, sizeof(uint32_t), 1, file);
    max_state_read = fread(&max_state, sizeof(int8_t), 1, file);
    min_state_read = fread(&min_state, sizeof(int8_t), 1, file);
    boost_true_positive_feedback_read = fread(&boost_true_positive_feedback, sizeof(uint8_t), 1, file);

    if (threshold_read != 1 || num_literals_read != 1 || num_clauses_read != 1 || num_classes_read != 1 ||
            max_state_read != 1 || min_state_read != 1 || boost_true_positive_feedback_read != 1) {
        fprintf(stderr, "Failed to read all metadata from bin\n");
        fclose(file);
        return NULL;
    }
    
    struct TsetlinMachine *tm = tm_create(
        num_classes, threshold, num_literals, num_clauses,
        max_state, min_state, boost_true_positive_feedback,
        y_size, y_element_size, s
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
        
        if (tm->pseudograd != NULL) {
            free(tm->pseudograd);
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
    
    for (uint32_t clause_id = 0; clause_id < tm->num_clauses; clause_id++) {
        for (uint32_t class_id = 0; class_id < tm->num_classes; class_id++) {
            tm->weights[(clause_id * tm->num_classes) + class_id] = 1;  // TODO: ?
        }
    }
}

// Translates automaton state to action - 0 or 1
static inline uint8_t action(int8_t state, int8_t mid_state) {
    return state >= mid_state;
}

// Calculate the output of each clause using the actions of each Tsetlin Automaton
// Output is stored an internal output array clause_output
static inline void calculate_clause_output(struct TsetlinMachine *tm, uint8_t *X) {
    uint8_t action_include, action_include_negated;
    uint8_t empty_clause;

    for (uint32_t clause_id = 0; clause_id < tm->num_clauses; clause_id++) {
        tm->clause_output[clause_id] = 1;
        empty_clause = 1;
        for (uint32_t literal_id = 0; literal_id < tm->num_literals; literal_id++) {
            action_include = action(tm->ta_state[(((clause_id * tm->num_literals) + literal_id) * 2) + 0], tm->mid_state);
            action_include_negated = action(tm->ta_state[(((clause_id * tm->num_literals) + literal_id) * 2) + 1], tm->mid_state);
            
            empty_clause = (empty_clause && (action_include || action_include_negated));

            if ((action_include == 1 && X[literal_id] == 0) || (action_include_negated == 1 && X[literal_id] == 1) || empty_clause) {
                tm->clause_output[clause_id] = 0;
                break;
            }
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
// Clause at clause_id voted correctly

// Type a - Clause is active for literals X (clause_output == 1)
static inline void type_1a_feedback(struct TsetlinMachine *tm, uint8_t *X, uint32_t clause_id) {
    // float s_inv = 1.0f / tm->s;
    // float s_min1_inv = (tm->s - 1.0f) / tm->s;

    for (uint32_t literal_id = 0; literal_id < tm->num_literals; literal_id++) {
        if (X[literal_id] == 1) {
            // True positive
            tm->ta_state[(((clause_id * tm->num_literals) + literal_id) * 2)] += 
                (tm->ta_state[(((clause_id * tm->num_literals) + literal_id) * 2)] < tm->max_state) && 
                (tm->boost_true_positive_feedback == 1 || 1.0*rand()/RAND_MAX <= tm->s_min1_inv);

            // False negative
            tm->ta_state[(((clause_id * tm->num_literals) + literal_id) * 2) + 1] -= 
                (tm->ta_state[(((clause_id * tm->num_literals) + literal_id) * 2) + 1] > tm->min_state) && 
                (1.0*rand()/RAND_MAX <= tm->s_inv);

        } else {
            // True negative
            tm->ta_state[(((clause_id * tm->num_literals) + literal_id) * 2) + 1] += 
                (tm->ta_state[(((clause_id * tm->num_literals) + literal_id) * 2) + 1] < tm->max_state) && 
                (1.0*rand()/RAND_MAX <= tm->s_min1_inv);
            
            // False positive
            tm->ta_state[(((clause_id * tm->num_literals) + literal_id) * 2)] -= 
                (tm->ta_state[(((clause_id * tm->num_literals) + literal_id) * 2)] > tm->min_state) && 
                (1.0*rand()/RAND_MAX <= tm->s_inv);
        }
    }
}


// Type b - Clause is inactive for literals X (clause_output == 0)
static inline void type_1b_feedback(struct TsetlinMachine *tm, uint32_t clause_id) {
    // float s_inv = 1.0f / tm->s;
    
    for (uint32_t literal_id = 0; literal_id < tm->num_literals; literal_id++) {
        tm->ta_state[(((clause_id * tm->num_literals) + literal_id) * 2)] -= 
            (tm->ta_state[(((clause_id * tm->num_literals) + literal_id) * 2)] > tm->min_state) && 
            (1.0*rand()/RAND_MAX <= tm->s_inv);
                            
        tm->ta_state[(((clause_id * tm->num_literals) + literal_id) * 2) + 1] -= 
            (tm->ta_state[(((clause_id * tm->num_literals) + literal_id) * 2) + 1] > tm->min_state) && 
            (1.0*rand()/RAND_MAX <= tm->s_inv);
    }
}


// Type II Feedback
// Clause at clause_id voted incorrectly
// && Clause is active for literals X (clause_output == 1)

static inline void type_2_feedback(struct TsetlinMachine *tm, uint8_t *X, uint32_t clause_id) {
    for (uint32_t literal_id = 0; literal_id < tm->num_literals; literal_id++) {
        tm->ta_state[(((clause_id * tm->num_literals) + literal_id) * 2)] += 
            0 == action(tm->ta_state[(((clause_id * tm->num_literals) + literal_id) * 2)], tm->mid_state) &&
            tm->ta_state[(((clause_id * tm->num_literals) + literal_id) * 2)] < tm->max_state &&
            0 == X[literal_id];

        tm->ta_state[(((clause_id * tm->num_literals) + literal_id) * 2) + 1] += 
            0 == action(tm->ta_state[(((clause_id * tm->num_literals) + literal_id) * 2) + 1], tm->mid_state) &&
            tm->ta_state[(((clause_id * tm->num_literals) + literal_id) * 2) + 1] < tm->max_state &&
            1 == X[literal_id];
    }
}


// The Tsetlin Machine can be trained incrementally, one training example at a time.
// Use this method directly for online and incremental training.
void tm_update(struct TsetlinMachine *tm, uint8_t *X, void *y) {

    calculate_clause_output(tm, X);

    for (uint32_t clause_id = 0; clause_id < tm->num_clauses; clause_id++) {
        // Calculate pseudo gradient - feedback to clauses
        int16_t *clause_votes_int16 = tm->weights + (clause_id * tm->num_classes);
        int8_t *pseudograd_row = tm->pseudograd + (clause_id * tm->num_classes);

        for (uint32_t class_id = 0; class_id < tm->num_classes; class_id++) {
            tm->votes[class_id] = (int32_t)clause_votes_int16[class_id];  // sign-extend into int32
        }

        tm->output_activation_pseudograd(tm, y, pseudograd_row);

        // Train Individual Automata
        for (uint32_t class_id = 0; class_id < tm->num_classes; class_id++) {
            if (tm->pseudograd[(clause_id * tm->num_classes) + class_id] > 0) {
                if (tm->clause_output[clause_id] == 1) {
                    type_1a_feedback(tm, X, clause_id);
                }
                else {
                    type_1b_feedback(tm, clause_id);
                }
            } else if (tm->pseudograd[(clause_id * tm->num_classes) + class_id] < 0 && tm->clause_output[clause_id] == 1) {
                type_2_feedback(tm, X, clause_id);
            }
        }
    }
}


// Inference
// y_pred should be allocated like: void *y_pred = malloc(rows * tm->y_size * tm->y_element_size);
void tm_score(struct TsetlinMachine *tm, uint8_t *X, void *y_pred, uint32_t rows) {
    for (uint32_t row = 0; row < rows; row++) {
        uint8_t* X_row = X + (row * tm->num_literals);
        void *y_pred_row = (void *)(((uint8_t *)y_pred) + (row * tm->y_size * tm->y_element_size));

        // Calculate clause output
        calculate_clause_output(tm, X_row);

        // Sum up clause votes for each class
        sum_votes(tm);

        // Pass through output activation function
        tm->output_activation(tm, y_pred_row);
    }
}


void tm_eval(struct TsetlinMachine *tm, uint8_t *X, void *y, uint32_t rows) {
    uint32_t correct = 0;
    uint32_t total = 0;
    void *y_pred = malloc(rows * tm->y_size * tm->y_element_size);
    if (y_pred == NULL) {
        perror("Memory allocation failed\n");
        exit(1);
    }

    tm_score(tm, X, y_pred, rows);
    
    for(uint32_t row = 0; row < rows; ++row) {

        void* y_row = (void *)(((uint8_t *)y) + (row * tm->y_size * tm->y_element_size));
        void* y_pred_row = (void *)(((uint8_t *)y_pred) + (row * tm->y_size * tm->y_element_size));
        
        if (tm->y_eq(tm, y_row, y_pred_row)) {
            correct++;
        }
        total++;
    }
    printf("correct: %d, total: %d, ratio: %.2f \n", correct, total, (float) correct / total);
}


void set_output_activation(
    struct TsetlinMachine *tm,
    void (*output_activation)(const struct TsetlinMachine *tm, void *y_pred)
) {
    tm->output_activation = output_activation;
}

void set_output_activation_pseudograd(
    struct TsetlinMachine *tm,
    void (*output_activation_pseudograd)(const struct TsetlinMachine *tm, const void *y, int8_t *pseudograd)
) {
    tm->output_activation_pseudograd = output_activation_pseudograd;
}


int8_t tm_get_state(struct TsetlinMachine *tm, uint32_t clause_id, uint32_t literal_id, uint8_t automaton_type) {
    return tm->ta_state[(((clause_id * tm->num_literals) + literal_id) * 2) + automaton_type];
}

int16_t tm_get_weight(struct TsetlinMachine *tm, uint32_t class_id, uint32_t clause_id) {
    return tm->weights[(clause_id * tm->num_classes) + class_id];
}
