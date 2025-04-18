#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tsetlin_machine.h"
#include "tm_output_activation.h"


// --- Basic y_eq function ---

uint8_t y_eq_generic(const struct TsetlinMachine *tm, const void *y, const void *y_pred) {
    return 0 == memcmp(y, y_pred, tm->y_size * tm->y_element_size);
}


// --- Tsetlin Machine ---

void tm_initialize(struct TsetlinMachine *tm);

/*** Initialize Tsetlin Machine ***/
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
    
    tm->clause_output = (int32_t *)malloc(num_clauses * sizeof(int32_t));  // shape: (num_clauses)
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

    /* Set up the Tsetlin Machine structure */

    tm_initialize(tm);
    
    return tm;
}


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

    // Read metadata
    fread(&threshold, sizeof(uint32_t), 1, file);
    fread(&num_literals, sizeof(uint32_t), 1, file);
    fread(&num_clauses, sizeof(uint32_t), 1, file);
    fread(&num_classes, sizeof(uint32_t), 1, file);
    fread(&max_state, sizeof(int8_t), 1, file);
    fread(&min_state, sizeof(int8_t), 1, file);
    fread(&boost_true_positive_feedback, sizeof(uint8_t), 1, file); 
    
    struct TsetlinMachine *tm = tm_create(
        num_classes, threshold, num_literals, num_clauses,
        max_state, min_state, boost_true_positive_feedback,
        y_size, y_element_size, s
    );
    if (!tm) {
        fprintf(stderr, "tm_create failed");
        fclose(file);
        return NULL;
    }

    // Allocate and read weights
    size_t weights_size = num_clauses * num_classes * sizeof(int16_t);
    fread(tm->weights, weights_size, 1, file);

    // Allocate and read clauses
    size_t clauses_size = num_clauses * num_literals * 2 * sizeof(int8_t);
    fread(tm->ta_state, clauses_size, 1, file);

    fclose(file);
    return tm;
}


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
        
        free(tm);
    }
    
    return;
}


void tm_initialize(struct TsetlinMachine *tm) {
    tm->mid_state = (tm->max_state + tm->min_state) / 2;

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

/* Translates automata state to action */
static inline uint8_t action(int state, int mid_state) {
    return state >= mid_state;
}

/* Calculate the output of each clause using the actions of each Tsetline Automaton. */
/* Output is stored an internal output array. */

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

// Sum up the votes for each class
static inline void sum_up_class_votes(struct TsetlinMachine *tm, int32_t *classes_sum) {
    memset((void *)classes_sum, 0, tm->num_classes*sizeof(int32_t));
    
    for (uint32_t clause_id = 0; clause_id < tm->num_clauses; clause_id++) {
        if (tm->clause_output[clause_id] == 0) {
            continue;
        }
        
        for (uint32_t class_id = 0; class_id < tm->num_classes; class_id++) {
            classes_sum[class_id] += tm->weights[(clause_id * tm->num_classes) + class_id];
        }
    }
    
    for (uint32_t class_id = 0; class_id < tm->num_classes; class_id++) {
        classes_sum[class_id] = (classes_sum[class_id] > (int32_t)tm->threshold) ? (int32_t)tm->threshold : classes_sum[class_id];
        classes_sum[class_id] = (classes_sum[class_id] < -(int32_t)tm->threshold) ? -(int32_t)tm->threshold : classes_sum[class_id];
    }
}

/*************************************************/
/*** Type I Feedback (Combats False Negatives) ***/
/*************************************************/

static inline void type_i_feedback(struct TsetlinMachine *tm, uint8_t *Xi, int clause_id, int class_id)
{
    // TODO: do this for class class_id with its weights
    /*
    if (tm->clause_output[j] == 0) {
        for (int k = 0; k < tm->num_literals; k++) {
            tm->ta_state[j][k][0] -= (tm->ta_state[j][k][0] > tm->min_state) && (1.0*rand()/RAND_MAX <= 1.0/s);
                                
            tm->ta_state[j][k][1] -= (tm->ta_state[j][k][1] > tm->min_state) && (1.0*rand()/RAND_MAX <= 1.0/s);
        }
    } else if (tm->clause_output[j] == 1) {
        for (int k = 0; k < tm->num_literals; k++) {
            if (Xi[k] == 1) {
                tm->ta_state[j][k][0] += (tm->ta_state[j][k][0] < tm->max_state) && (tm->boost_true_positive_feedback == 1 || 1.0*rand()/RAND_MAX <= (s-1)/s);

                tm->ta_state[j][k][1] -= (tm->ta_state[j][k][1] > tm->min_state) && (1.0*rand()/RAND_MAX <= 1.0/s);
            } else if (Xi[k] == 0) {
                tm->ta_state[j][k][1] += (tm->ta_state[j][k][1] < tm->max_state) && (tm->boost_true_positive_feedback == 1 || 1.0*rand()/RAND_MAX <= (s-1)/s);
                
                tm->ta_state[j][k][0] -= (tm->ta_state[j][k][0] > tm->min_state) && (1.0*rand()/RAND_MAX <= 1.0/s);
            }
        }
    }
    */
}


/**************************************************/
/*** Type II Feedback (Combats False Positives) ***/
/**************************************************/

static inline void type_ii_feedback(struct TsetlinMachine *tm, uint8_t *Xi, int clause_id, int class_id) {
    // TODO: do this for class class_id with its weights
    /*
    int action_include;
    int action_include_negated;

    if (tm->clause_output[j] == 1) {
        for (int k = 0; k < tm->num_literals; k++) { 
            action_include = action(tm->ta_state[j][k][0], tm->mid_state);
            action_include_negated = action(tm->ta_state[j][k][1], tm->mid_state);

            tm->ta_state[j][k][0] += (action_include == 0 && tm->ta_state[j][k][0] < tm->max_state) && (Xi[k] == 0);
            tm->ta_state[j][k][1] += (action_include_negated == 0 && tm->ta_state[j][k][1] < tm->max_state) && (Xi[k] == 1);
        }
    }
    */
}

// The Tsetlin Machine can be trained incrementally, one training example at a time.
// Use this method directly for online and incremental training.
// Slow due to malloc, prefer batching data when possible
void tm_update(struct TsetlinMachine *tm, uint8_t *X, void *y) {
    int32_t *clause_votes = malloc(tm->num_classes * sizeof(int32_t));
    if (clause_votes == NULL) {
        perror("Memory allocation failed\n");
        exit(1);
    }
    calculate_clause_output(tm, X);

    for (uint32_t clause_id = 0; clause_id < tm->num_clauses; clause_id++) {
        if (tm->clause_output[clause_id] == 0) {
            continue;
        }

        // Calculate pseudo gradient - feedback to clauses
        int16_t *clause_votes_int16 = tm->weights + (clause_id * tm->num_classes);
        int8_t *pseudograd_row = tm->pseudograd + (clause_id * tm->num_classes);

        for (uint32_t class_id = 0; class_id < tm->num_classes; class_id++) {
            clause_votes[class_id] = (int32_t)clause_votes_int16[class_id];  // sign-extend into int32
        }

        tm->output_activation_pseudograd(tm, y, clause_votes, pseudograd_row);

        // Train Individual Automata
        for (uint32_t class_id = 0; class_id < tm->num_classes; class_id++) {
            if (tm->pseudograd[(clause_id * tm->num_classes) + class_id] > 0) {
                type_i_feedback(tm, X, class_id, clause_id);
            } else if (tm->pseudograd[(clause_id * tm->num_classes) + class_id] < 0) {
                type_ii_feedback(tm, X, class_id, clause_id);
            }
        }
    }
}

void tm_score(struct TsetlinMachine *tm, uint8_t *X, int32_t *votes) {
    // Calculate clause output
    calculate_clause_output(tm, X);

    // Sum up clause votes
    sum_up_class_votes(tm, votes);
}

void tm_eval(struct TsetlinMachine *tm, uint8_t *X, void *y, uint32_t rows, uint32_t cols) {
    uint32_t correct = 0;
    uint32_t total = 0;
    int32_t *votes = malloc(tm->num_classes * sizeof(int32_t));
    void *y_pred = malloc(tm->y_size * tm->y_element_size);
    if (votes == NULL || y_pred == NULL) {
        perror("Memory allocation failed\n");
        exit(1);
    }
    
    for(uint32_t row = 0; row < rows; ++row) {
        if (row % 1000 == 0 && row) {
            printf("%d out of %d done\n", row, rows);
        }

        uint8_t* X_row = X + (row * cols);
        void* y_row = (void *)(((uint8_t *)y) + (row * tm->y_size * tm->y_element_size));

        tm_score(tm, X_row, votes);
        
        tm->output_activation(tm, votes, y_pred);
        
        if (tm->y_eq(tm, y_row, y_pred)) {
            correct++;
        }
        total++;
    }
    printf("correct: %d, total: %d, ratio: %.2f \n", correct, total, (float) correct / total);
}

int8_t tm_get_state(struct TsetlinMachine *tm, uint32_t clause_id, uint32_t literal_id, uint8_t automaton_type) {
    return tm->ta_state[(((clause_id * tm->num_literals) + literal_id) * 2) + automaton_type];
}

int16_t tm_get_weight(struct TsetlinMachine *tm, uint32_t class_id, uint32_t clause_id) {
    return tm->weights[(clause_id * tm->num_classes) + class_id];
}

