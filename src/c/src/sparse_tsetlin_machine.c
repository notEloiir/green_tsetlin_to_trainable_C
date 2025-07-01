#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#include "sparse_tsetlin_machine.h"
#include "utility.h"


// Insert a new node into a linked list after prev
// If prev is NULL, insert at the head of the list
// If head_ptr is NULL, it will be initialized to the new node
// If result is not NULL, it will point to the new node
void ta_state_insert(struct TAStateNode **head_ptr, struct TAStateNode *prev, uint32_t ta_id, uint8_t ta_state, struct TAStateNode **result) {
	struct TAStateNode *node = malloc(sizeof(struct TAStateNode));
	if (node == NULL) {
		perror("Memory allocation failed");
		exit(1);
	}
	node->ta_id = ta_id;
	node->ta_state = ta_state;
	node->next = prev != NULL ? prev->next : NULL;

	if (*head_ptr == NULL) {
		*head_ptr = node;
	}
	else if (prev == NULL) {
		node->next = *head_ptr;
		*head_ptr = node;
	}
	else {
		prev->next = node;
	}

	if (result != NULL) {
		*result = node;
	}
}

// Remove a node from a linked list after prev
// If prev is NULL, remove the head of the list
// If head_ptr is NULL, it will not be modified
// If result is not NULL, it will point to the next node after the removed node
void ta_state_remove(struct TAStateNode **head_ptr, struct TAStateNode *prev, struct TAStateNode **result) {
	if (*head_ptr == NULL) {
        fprintf(stderr, "Trying to remove from empty linked list\n");
        return;
	}
	if (prev != NULL && prev->next == NULL) {
        // Trying to remove node after tail of linked list
        return;
	}

	struct TAStateNode *to_remove = NULL;
	if (prev == NULL) {
		// Removing first element
		to_remove = *head_ptr;
		*head_ptr = to_remove->next;
	}
	else {
		to_remove = prev->next;
		prev->next = to_remove->next;
	}

	if (result != NULL) {
		*result = to_remove->next;
	}

	free(to_remove);
}


// --- Basic y_eq function ---

uint8_t stm_y_eq_generic(const struct SparseTsetlinMachine *stm, const void *y, const void *y_pred) {
    return 0 == memcmp(y, y_pred, stm->y_size * stm->y_element_size);
}


// --- Tsetlin Machine ---

void stm_initialize(struct SparseTsetlinMachine *stm);
static inline void stm_clear_llists(struct SparseTsetlinMachine *stm);

// Translates automaton state to action - 0 or 1
static inline uint8_t action(int8_t state, int8_t mid_state) {
    return state >= mid_state;
}

// Allocate memory, fill in fields, calls stm_initialize
struct SparseTsetlinMachine *stm_create(
    uint32_t num_classes, uint32_t threshold, uint32_t num_literals, uint32_t num_clauses,
    int8_t max_state, int8_t min_state, uint8_t boost_true_positive_feedback,
    uint32_t y_size, uint32_t y_element_size, float s, uint32_t seed
) {
    struct SparseTsetlinMachine *stm = (struct SparseTsetlinMachine *)malloc(sizeof(struct SparseTsetlinMachine));
    if(stm == NULL) {
        perror("Memory allocation failed");
        return NULL;
    }
    
    stm->num_classes = num_classes;
    stm->threshold = threshold;
    stm->num_literals = num_literals;
    stm->num_clauses = num_clauses;
    stm->max_state = max_state;
    stm->min_state = min_state;
    stm->boost_true_positive_feedback = boost_true_positive_feedback;
    stm->s = s;
    
    stm->y_size = y_size;
    stm->y_element_size = y_element_size;
    stm->y_eq = stm_y_eq_generic;
    stm->output_activation = stm_oa_class_idx;
    stm->calculate_feedback = stm_feedback_class_idx;

    stm->ta_state = (struct TAStateNode **)malloc(num_clauses * sizeof(struct TAStateNode *));  // shape: flat (num_clauses)
    if (stm->ta_state == NULL) {
        perror("Memory allocation failed");
        stm_free(stm);
        return NULL;
    }
    for (uint32_t clause_id = 0; clause_id < stm->num_clauses; clause_id++) {
    	stm->ta_state[clause_id] = NULL;
    }

    stm->active_literals = (struct TANode **)malloc(num_classes * sizeof(struct TANode *));  // shape: flat (num_classes)
    if (stm->active_literals == NULL) {
        perror("Memory allocation failed");
        stm_free(stm);
        return NULL;
    }
    for (uint32_t class_id = 0; class_id < stm->num_classes; class_id++) {
    	stm->active_literals[class_id] = NULL;
    }
    
    stm->weights = (int16_t *)malloc(num_clauses * num_classes * sizeof(int16_t));  // shape: flat (num_clauses, num_classes)
    if (stm->weights == NULL) {
        perror("Memory allocation failed");
        stm_free(stm);
        return NULL;
    }
    
    stm->clause_output = (uint8_t *)malloc(num_clauses * sizeof(uint8_t));  // shape: (num_clauses)
    if (stm->clause_output == NULL) {
        perror("Memory allocation failed");
        stm_free(stm);
        return NULL;
    }

    stm->votes = (int32_t *)malloc(num_classes * sizeof(int32_t));  // shape: (num_classes)
    if (stm->votes == NULL) {
        perror("Memory allocation failed");
        stm_free(stm);
        return NULL;
    }

    prng_seed(&(stm->rng), seed);

    /* Set up the Tsetlin Machine structure */

    stm_initialize(stm);
    
    return stm;
}


// Load Tsetlin Machine from a bin file
struct SparseTsetlinMachine *stm_load_dense(
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
    uint32_t seed = 42;

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
    
    struct SparseTsetlinMachine *stm = stm_create(
        num_classes, threshold, num_literals, num_clauses,
        max_state, min_state, boost_true_positive_feedback,
        y_size, y_element_size, (float)s_double, seed
    );
    if (!stm) {
        fprintf(stderr, "stm_create failed\n");
        fclose(file);
        return NULL;
    }

    // Allocate and read weights
    size_t weights_read = fread(stm->weights, sizeof(int16_t), num_clauses * num_classes, file);
    if (weights_read != num_clauses * num_classes) {
        fprintf(stderr, "Failed to read all weights from bin\n");
        stm_free(stm);
        fclose(file);
        return NULL;
    }
    // Allocate and read clauses
    int8_t *flat_states = malloc(num_clauses * num_literals * 2 * sizeof(int8_t));
    size_t states_read = fread(flat_states, sizeof(int8_t), num_clauses * num_literals * 2, file);
    if (states_read != num_clauses * num_literals * 2) {
        fprintf(stderr, "Failed to read all states from bin\n");
        stm_free(stm);
        free(flat_states);
        fclose(file);
        return NULL;
    }

	stm_clear_llists(stm);
    for (uint32_t clause_id = 0; clause_id < stm->num_clauses; clause_id++) {
    	struct TAStateNode *prev_ptr = NULL;
    	struct TAStateNode **head_ptr_addr = stm->ta_state + clause_id;

        for (uint32_t i = 0; i < stm->num_literals * 2; i++) {
        	if (action(flat_states[clause_id * stm->num_literals * 2 + i], stm->mid_state)) {
        		ta_state_insert(head_ptr_addr, prev_ptr, i, flat_states[clause_id * stm->num_literals * 2 + i], &prev_ptr);
        	}
        }
    }
    free(flat_states);

    fclose(file);
    return stm;
}


void stm_save(const struct SparseTsetlinMachine *stm, const char *filename) {
    FILE *file = fopen(filename, "wb");
    if (!file) {
        perror("Error opening file for writing");
        return;
    }

    size_t written;

    written = fwrite(&stm->threshold, sizeof(uint32_t), 1, file);
    if (written != 1) {
        fprintf(stderr, "Failed to write threshold\n");
        goto save_error;
    }
    written = fwrite(&stm->num_literals, sizeof(uint32_t), 1, file);
    if (written != 1) {
        fprintf(stderr, "Failed to write num_literals\n");
        goto save_error;
    }
    written = fwrite(&stm->num_clauses, sizeof(uint32_t), 1, file);
    if (written != 1) {
        fprintf(stderr, "Failed to write num_clauses\n");
        goto save_error;
    }
    written = fwrite(&stm->num_classes, sizeof(uint32_t), 1, file);
    if (written != 1) {
        fprintf(stderr, "Failed to write num_classes\n");
        goto save_error;
    }
    written = fwrite(&stm->max_state, sizeof(int8_t), 1, file);
    if (written != 1) {
        fprintf(stderr, "Failed to write max_state\n");
        goto save_error;
    }
    written = fwrite(&stm->min_state, sizeof(int8_t), 1, file);
    if (written != 1) {
        fprintf(stderr, "Failed to write min_state\n");
        goto save_error;
    }
    written = fwrite(&stm->boost_true_positive_feedback, sizeof(uint8_t), 1, file);
    if (written != 1) {
        fprintf(stderr, "Failed to write boost_true_positive_feedback\n");
        goto save_error;
    }
    written = fwrite(&stm->s, sizeof(double), 1, file);
    if (written != 1) {
        fprintf(stderr, "Failed to write s parameter\n");
        goto save_error;
    }
    size_t n_weights = (size_t)stm->num_clauses * stm->num_classes;
    written = fwrite(stm->weights, sizeof(int16_t), n_weights, file);
    if (written != n_weights) {
        fprintf(stderr, "Failed to write weights array (%zu of %zu)\n",
                written, n_weights);
        goto save_error;
    }
    for (uint32_t clause_id = 0; clause_id < stm->num_clauses; clause_id++) {
    	struct TAStateNode *curr_ptr = stm->ta_state[clause_id];
    	while (curr_ptr != NULL) {
    		written = fwrite(&curr_ptr->ta_id, sizeof(uint32_t), 1, file);
			if (written != 1) {
				fprintf(stderr, "Failed to write node ta_id\n");
				goto save_error;
    		}
    		written = fwrite(&curr_ptr->ta_state, sizeof(int8_t), 1, file);
			if (written != 1) {
				fprintf(stderr, "Failed to write node ta_state\n");
				goto save_error;
    		}
			curr_ptr = curr_ptr->next;
    	}
    	uint32_t delim = UINT_MAX;
		written = fwrite(&delim, sizeof(uint32_t), 1, file);
		if (written != 1) {
			fprintf(stderr, "Failed to write delimiter\n");
			goto save_error;
		}
    }

    fclose(file);
    return;

save_error:
    fclose(file);
    fprintf(stderr, "stm_save aborted, file %s may be incomplete\n", filename);
}


static inline void stm_clear_llists(struct SparseTsetlinMachine *stm) {
	for (uint32_t clause_id = 0; clause_id < stm->num_clauses; clause_id++) {
		struct TAStateNode **head_ptr = stm->ta_state + clause_id;
		while (*head_ptr != NULL) {
			ta_state_remove(head_ptr, NULL, NULL);
		}
    }

    for (uint32_t class_id = 0; class_id < stm->num_classes; class_id++) {
		struct TANode **al_head_ptr = stm->active_literals + class_id;
		while (*al_head_ptr != NULL) {
			ta_stateless_remove(al_head_ptr, NULL, NULL);
		}
	}
}

// Free all allocated memory
void stm_free(struct SparseTsetlinMachine *stm) {
    if (stm != NULL){
    	if (stm->ta_state != NULL || stm->active_literals != NULL) {
            stm_clear_llists(stm);

            free(stm->ta_state);
            free(stm->active_literals);

            stm->ta_state = NULL;
            stm->active_literals = NULL;
		}
        
        if (stm->weights != NULL) {
            free(stm->weights);
            stm->weights = NULL;
        }
        
        if (stm->clause_output != NULL) {
            free(stm->clause_output);
            stm->clause_output = NULL;
        }
        
        if (stm->votes != NULL) {
            free(stm->votes);
            stm->votes = NULL;
        }
        
        free(stm);
    }
    
    return;
}


// Initialize values
void stm_initialize(struct SparseTsetlinMachine *stm) {
    stm->mid_state = (stm->max_state + stm->min_state) / 2;
    stm->sparse_init_state = stm->mid_state - 35;
    stm->sparse_min_state = stm->mid_state - 40;
    stm->s_inv = 1.0f / stm->s;
    stm->s_min1_inv = (stm->s - 1.0f) / stm->s;
    
    // Init weights randomly to -1 or 1
    for (uint32_t clause_id = 0; clause_id < stm->num_clauses; clause_id++) {
        for (uint32_t class_id = 0; class_id < stm->num_classes; class_id++) {
            stm->weights[(clause_id * stm->num_classes) + class_id] = 1 - 2*(prng_next_float(&(stm->rng)) <= 0.5);
        }
    }
}

// Calculate the output of each clause using the actions of each Tsetlin Automaton
// Output is stored an internal output array clause_output
static inline void calculate_clause_output(struct SparseTsetlinMachine *stm, const uint8_t *X, uint8_t skip_empty) {
    // For each clause, check if it is "active" - all necessary literals have the right value
    for (uint32_t clause_id = 0; clause_id < stm->num_clauses; clause_id++) {
        stm->clause_output[clause_id] = 1;
        uint8_t empty_clause = 1;

		struct TAStateNode *curr_ptr = stm->ta_state[clause_id];
		while (curr_ptr != NULL) {
			if (action(curr_ptr->ta_state, stm->mid_state)) {
				empty_clause = 0;
				if (curr_ptr->ta_id % 2 == X[curr_ptr->ta_id / 2]) {
					stm->clause_output[clause_id] = 0;
					break;
				}
			}
			curr_ptr = curr_ptr->next;
		}
		if (empty_clause && skip_empty) {
			stm->clause_output[clause_id] = 0;
		}
    }
}


// Sum up the votes of each clause for each class
static inline void sum_votes(struct SparseTsetlinMachine *stm) {
    memset(stm->votes, 0, stm->num_classes*sizeof(int32_t));
    
    for (uint32_t clause_id = 0; clause_id < stm->num_clauses; clause_id++) {
        if (stm->clause_output[clause_id] == 0) {
            continue;
        }
        
        for (uint32_t class_id = 0; class_id < stm->num_classes; class_id++) {
            stm->votes[class_id] += stm->weights[(clause_id * stm->num_classes) + class_id];
        }
    }
    
    for (uint32_t class_id = 0; class_id < stm->num_classes; class_id++) {
        stm->votes[class_id] = clip(stm->votes[class_id], (int32_t)stm->threshold);
    }
}


// Type I Feedback
// Clause at clause_id voted correctly for class at class_id

// Type a - Clause is active for literals X (clause_output == 1)
void type_1a_feedback(struct SparseTsetlinMachine *stm, const uint8_t *X, uint32_t clause_id, uint32_t class_id) {
    // float s_inv = 1.0f / stm->s;
    // float s_min1_inv = (stm->s - 1.0f) / stm->s;

    uint8_t feedback_strength = 1;

    if (stm->weights[clause_id * stm->num_classes + class_id] >= 0) {
        stm->weights[clause_id * stm->num_classes + class_id] += min(feedback_strength, SHRT_MAX - stm->weights[clause_id * stm->num_classes + class_id]);
    }
    else {
        stm->weights[clause_id * stm->num_classes + class_id] -= min(feedback_strength, -(SHRT_MIN - stm->weights[clause_id * stm->num_classes + class_id]));
    }
    
    struct TAStateNode *state_ptr = stm->ta_state[clause_id];
    struct TAStateNode *prev_state_ptr = NULL;

    struct TANode *al_ptr = stm->active_literals[class_id];
    struct TANode *prev_al_ptr = NULL;

    for (uint32_t i = 0; i < stm->num_literals * 2; i++) {
        uint8_t is_active_literal = al_ptr != NULL && al_ptr->ta_id == i / 2;
        if (is_active_literal && i % 2 == 1) {
            prev_al_ptr = al_ptr;
            al_ptr = al_ptr->next;
        }

    	if (state_ptr == NULL || state_ptr->ta_id != i) {
    		if (!is_active_literal && i % 2 == 0 && X[i / 2] == 1) {
                // (i % 2 != X[i / 2]) means TA i "votes" correctly (condition for applying 1a feedback)
                // (i % 2 == 0) means only positive TAs
                // (X[i / 2] == 1) means literal at i/2 is active

				// Insert new active literal i/2 for class class_id
				ta_stateless_insert(stm->active_literals + class_id, prev_al_ptr, i / 2, &prev_al_ptr);
				al_ptr = prev_al_ptr->next;
    		}
    		continue;
    	}

        // X[i / 2] should equal action at ta_id==i
        if (state_ptr->ta_id % 2 != X[state_ptr->ta_id / 2]) {
            // Correct, reward
            state_ptr->ta_state +=
				min(stm->max_state - state_ptr->ta_state, feedback_strength) *
				(stm->boost_true_positive_feedback == 1 || prng_next_float(&(stm->rng)) <= stm->s_min1_inv);
        }
        else {
            // Incorrect, punish
            state_ptr->ta_state -=
				min(-(stm->min_state - state_ptr->ta_state), feedback_strength) *
				prng_next_float(&(stm->rng)) <= stm->s_inv;

            if (state_ptr->ta_state < stm->sparse_min_state) {
            	// Remove TA
            	ta_state_remove(stm->ta_state + clause_id, prev_state_ptr, &state_ptr);
                continue;
            }
        }

        prev_state_ptr = state_ptr;
        state_ptr = state_ptr->next;
    }
}


// Type b - Clause is inactive for literals X (clause_output == 0)
void type_1b_feedback(struct SparseTsetlinMachine *stm, uint32_t clause_id) {
    // float s_inv = 1.0f / stm->s;

    uint8_t feedback_strength = 1;

    struct TAStateNode *state_ptr = stm->ta_state[clause_id];
    struct TAStateNode *prev_state_ptr = NULL;
    for (uint32_t i = 0; i < stm->num_literals * 2; i++) {
    	if (state_ptr == NULL || state_ptr->ta_id != i) {
    		continue;
    	}

        state_ptr->ta_state -=
			min(-(stm->min_state - state_ptr->ta_state), feedback_strength) *
			prng_next_float(&(stm->rng)) <= stm->s_inv;

        if (state_ptr->ta_state < stm->sparse_min_state) {
        	// Remove TA
        	ta_state_remove(stm->ta_state + clause_id, prev_state_ptr, &state_ptr);
        	continue;
        }

        prev_state_ptr = state_ptr;
        state_ptr = state_ptr->next;
    }
}


// Type II Feedback
// Clause at clause_id voted incorrectly for class at class_id
// && Clause is active for literals X (clause_output == 1)

void type_2_feedback(struct SparseTsetlinMachine *stm, const uint8_t *X, uint32_t clause_id, uint32_t class_id) {
    uint8_t feedback_strength = 1;

    stm->weights[clause_id * stm->num_classes + class_id] +=
        stm->weights[clause_id * stm->num_classes + class_id] >= 0 ? -feedback_strength : feedback_strength;

    struct TAStateNode *state_ptr = stm->ta_state[clause_id];
    struct TAStateNode *prev_state_ptr = NULL;

    struct TANode *al_ptr = stm->active_literals[class_id];

    for (uint32_t i = 0; i < stm->num_literals * 2; i++) {
        uint8_t is_active_literal = al_ptr != NULL && al_ptr->ta_id == i / 2;
        if (is_active_literal && i % 2 == 1) {
            al_ptr = al_ptr->next;
        }

    	if (state_ptr == NULL || state_ptr->ta_id != i) {
            if (is_active_literal && (i % 2 == 0 || (i % 2 == 1 && X[i / 2] == 1))) {
                // (i % 2 == X[i / 2]) means TA i "votes" incorrectly (condition for applying 2 feedback)

				// Insert new TA with state stm->mid_state
				ta_state_insert(stm->ta_state + clause_id, prev_state_ptr, i, stm->sparse_init_state, &prev_state_ptr);
				state_ptr = prev_state_ptr->next;
    		}
    		continue;
    	}

        state_ptr->ta_state +=
            min(stm->max_state - state_ptr->ta_state, feedback_strength) * (
            0 == action(state_ptr->ta_state, stm->mid_state) &&
            i % 2 == X[i / 2]);

        prev_state_ptr = state_ptr;
        state_ptr = state_ptr->next;
    }
}


void stm_train(struct SparseTsetlinMachine *stm, const uint8_t *X, const void *y, uint32_t rows, uint32_t epochs) {
    for (uint32_t epoch = 0; epoch < epochs; epoch++) {
		for (uint32_t row = 0; row < rows; row++) {
			const uint8_t *X_row = X + (row * stm->num_literals);
			void *y_row = (void *)((uint8_t *)y + (row * stm->y_size * stm->y_element_size));

			calculate_clause_output(stm, X_row, 0);

			sum_votes(stm);

			// Calculate and apply feedback to all clauses
			stm->calculate_feedback(stm, X_row, y_row);
		}
    }
}


// Inference
// y_pred should be allocated like: void *y_pred = malloc(rows * stm->y_size * stm->y_element_size);
void stm_predict(struct SparseTsetlinMachine *stm, const uint8_t *X, void *y_pred, uint32_t rows) {
    for (uint32_t row = 0; row < rows; row++) {
    	const uint8_t* X_row = X + (row * stm->num_literals);
        void *y_pred_row = (void *)(((uint8_t *)y_pred) + (row * stm->y_size * stm->y_element_size));

        // Calculate clause output
        calculate_clause_output(stm, X_row, 1);

        // Sum up clause votes for each class
        sum_votes(stm);

        // Pass through output activation function
        stm->output_activation(stm, y_pred_row);
    }
}


void stm_evaluate(struct SparseTsetlinMachine *stm, const uint8_t *X, const void *y, uint32_t rows) {
    uint32_t correct = 0;
    uint32_t total = 0;
    void *y_pred = malloc(rows * stm->y_size * stm->y_element_size);
    if (y_pred == NULL) {
        perror("Memory allocation failed\n");
        exit(1);
    }

    stm_predict(stm, X, y_pred, rows);
    
    for(uint32_t row = 0; row < rows; ++row) {

        void* y_row = (void *)(((uint8_t *)y) + (row * stm->y_size * stm->y_element_size));
        void* y_pred_row = (void *)(((uint8_t *)y_pred) + (row * stm->y_size * stm->y_element_size));
        
        if (stm->y_eq(stm, y_row, y_pred_row)) {
            correct++;
        }
        total++;
    }
    printf("correct: %d, total: %d, ratio: %.2f \n", correct, total, (float) correct / total);
    free(y_pred);
}


// --- Basic output_activation functions ---

void stm_oa_class_idx(const struct SparseTsetlinMachine *stm, const void *y_pred) {
    if (stm->y_size != 1) {
        fprintf(stderr, "y_eq_class_idx expects y_size == 1");
        exit(1);
    }
    uint32_t *label_pred = (uint32_t *)y_pred;

    // class index compare
    uint32_t best_class = 0;
    int32_t max_class_score = stm->votes[0];
    for (uint32_t class_id = 1; class_id < stm->num_classes; class_id++) {
        if (max_class_score < stm->votes[class_id]) {
            max_class_score = stm->votes[class_id];
            best_class = class_id;
        }
    }

    *label_pred = best_class;
}

void stm_oa_bin_vector(const struct SparseTsetlinMachine *stm, const void *y_pred) {
    if(stm->y_size != stm->num_classes) {
        fprintf(stderr, "y_eq_bin_vector expects y_size == tm->num_classes");
        exit(1);
    }
    uint8_t *y_bin_vec = (uint8_t *)y_pred;

    for (uint32_t class_id = 0; class_id < stm->num_classes; class_id++) {
        // binary threshold (k=mid_state)
        y_bin_vec[class_id] = (stm->votes[class_id] > stm->mid_state);
    }
}


void stm_set_output_activation(
    struct SparseTsetlinMachine *stm,
    void (*output_activation)(const struct SparseTsetlinMachine *stm, const void *y_pred)
) {
    stm->output_activation = output_activation;
}


// Internal component of feedback functions below
void stm_apply_feedback(struct SparseTsetlinMachine *stm, uint32_t clause_id, uint32_t class_id, uint8_t is_class_positive, const uint8_t *X) {
	uint8_t is_vote_positive = stm->weights[(clause_id * stm->num_classes) + class_id] >= 0;
	if (is_vote_positive == is_class_positive) {
		if (stm->clause_output[clause_id] == 1) {
			type_1a_feedback(stm, X, clause_id, class_id);
		}
		else {
			type_1b_feedback(stm, clause_id);
		}
	}
	else if (stm->clause_output[clause_id] == 1) {
		type_2_feedback(stm, X, clause_id, class_id);
	}
}

// --- calculate_feedback ---
// Calculate clause-class feedback

void stm_feedback_class_idx(struct SparseTsetlinMachine *stm, const uint8_t *X, const void *y) {
    // Correct label gets feedback type 1a or 1b, incorrect maybe get type 2 (depending on clause output)
    const uint32_t *label_ptr = (const uint32_t *)y;
    const uint32_t positive_class = *label_ptr;
    uint32_t negative_class = 0;

    int32_t votes_clipped_positive = clip(stm->votes[positive_class], (int32_t)stm->threshold);
	float update_probability_positive = ((float)stm->threshold - (float)votes_clipped_positive) / (float)(2 * stm->threshold);

	for (uint32_t clause_id = 0; clause_id < stm->num_clauses; clause_id++) {
		if (prng_next_float(&(stm->rng)) <= update_probability_positive) {
			stm_apply_feedback(stm, clause_id, positive_class, 1, X);
		}
	}

    int32_t sum_votes_clipped_negative = 0;
    for (uint32_t class_id = 0; class_id < stm->num_classes; class_id++) {
        if (class_id != positive_class) {
            sum_votes_clipped_negative += clip(stm->votes[class_id], (int32_t)stm->threshold) + (int32_t)stm->threshold;
        }
    }
    if (sum_votes_clipped_negative == 0) return;
    int32_t random_vote_negative = prng_next_uint32(&(stm->rng)) % sum_votes_clipped_negative;
    int32_t accumulated_votes = 0;
    for (uint32_t class_id = 0; class_id < stm->num_classes; class_id++) {
        if (class_id != positive_class) {
            accumulated_votes += clip(stm->votes[class_id], (int32_t)stm->threshold) + (int32_t)stm->threshold;
            if (accumulated_votes >= random_vote_negative) {
                negative_class = class_id;
                break;
            }
        }
    }

    int32_t votes_clipped_negative = clip(stm->votes[negative_class], (int32_t)stm->threshold);
    float update_probability_negative = ((float)votes_clipped_negative + (float)stm->threshold) / (float)(2 * stm->threshold);

    for (uint32_t clause_id = 0; clause_id < stm->num_clauses; clause_id++) {
		if (prng_next_float(&(stm->rng)) <= update_probability_negative) {
			stm_apply_feedback(stm, clause_id, negative_class, 0, X);
		}
    }
}

void stm_feedback_bin_vector(struct SparseTsetlinMachine *stm, const uint8_t *X, const void *y) {
    const uint8_t *label_arr = (const uint8_t *)y;
    uint32_t positive_class = 0;
    uint32_t negative_class = 0;

    int32_t sum_votes_clipped_positive = 0;
	for (uint32_t class_id = 0; class_id < stm->num_classes; class_id++) {
		if (label_arr[class_id]) {
			sum_votes_clipped_positive += clip(stm->votes[class_id], (int32_t)stm->threshold) + (int32_t)stm->threshold;
		}
	}
	if (sum_votes_clipped_positive == 0) goto negative_feedback;
	int32_t random_vote_positive = prng_next_uint32(&(stm->rng)) % sum_votes_clipped_positive;
	int32_t accumulated_votes_positive = 0;
	for (uint32_t class_id = 0; class_id < stm->num_classes; class_id++) {
		if (label_arr[class_id]) {
			accumulated_votes_positive += clip(stm->votes[class_id], (int32_t)stm->threshold) + (int32_t)stm->threshold;
			if (accumulated_votes_positive >= random_vote_positive) {
				positive_class = class_id;
				break;
			}
		}
	}

	int32_t votes_clipped_positive = clip(stm->votes[negative_class], (int32_t)stm->threshold);
	float update_probability_positive = ((float)stm->threshold - (float)votes_clipped_positive) / (float)(2 * stm->threshold);

	for (uint32_t clause_id = 0; clause_id < stm->num_clauses; clause_id++) {
		if (prng_next_float(&(stm->rng)) <= update_probability_positive) {
			stm_apply_feedback(stm, clause_id, positive_class, 1, X);
		}
	}

negative_feedback:

    int32_t sum_votes_clipped_negative = 0;
	for (uint32_t class_id = 0; class_id < stm->num_classes; class_id++) {
		if (!label_arr[class_id]) {
			sum_votes_clipped_negative += clip(stm->votes[class_id], (int32_t)stm->threshold) + (int32_t)stm->threshold;
		}
	}
	if (sum_votes_clipped_negative == 0) return;
	int32_t random_vote_negative = prng_next_uint32(&(stm->rng)) % sum_votes_clipped_negative;
	int32_t accumulated_votes_negative = 0;
	for (uint32_t class_id = 0; class_id < stm->num_classes; class_id++) {
		if (!label_arr[class_id]) {
			accumulated_votes_negative += clip(stm->votes[class_id], (int32_t)stm->threshold) + (int32_t)stm->threshold;
			if (accumulated_votes_negative >= random_vote_negative) {
				negative_class = class_id;
				break;
			}
		}
	}

	int32_t votes_clipped_negative = clip(stm->votes[negative_class], (int32_t)stm->threshold);
	float update_probability_negative = ((float)votes_clipped_negative + (float)stm->threshold) / (float)(2 * stm->threshold);

	for (uint32_t clause_id = 0; clause_id < stm->num_clauses; clause_id++) {
		if (prng_next_float(&(stm->rng)) <= update_probability_negative) {
			stm_apply_feedback(stm, clause_id, negative_class, 0, X);
		}
	}
}


void stm_set_calculate_feedback(
    struct SparseTsetlinMachine *stm,
    void (*calculate_feedback)(struct SparseTsetlinMachine *stm, const uint8_t *X, const void *y)
) {
    stm->calculate_feedback = calculate_feedback;
}
