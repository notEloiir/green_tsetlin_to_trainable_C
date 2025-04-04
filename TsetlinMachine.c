/*
(Heavily) Modified code from https://github.com/cair/TsetlinMachineC
*/

/*

Copyright (c) 2019 Ole-Christoffer Granmo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

This code implements the Tsetlin Machine from paper arXiv:1804.01508
https://arxiv.org/abs/1804.01508

*/

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "TsetlinMachine.h"

void tm_initialize(struct TsetlinMachine *tm);

/**************************************/
/*** The Multiclass Tsetlin Machine ***/
/**************************************/

/*** Initialize Tsetlin Machine ***/
struct TsetlinMachine *create_tsetlin_machine(int n_classes, int threshold, int n_literals, int n_clauses, int max_state, int min_state, int boost_true_positive_feedback, int predict, int update)
{
	struct TsetlinMachine *tm = (struct TsetlinMachine *)malloc(sizeof(struct TsetlinMachine));
	if(tm == NULL) {
		perror("Memory allocation failed");
		return NULL;
	}
	
	tm->n_classes = n_classes;
	tm->threshold = threshold;
	tm->n_literals = n_literals;
	tm->n_clauses = n_clauses;
	tm->max_state = max_state;
	tm->min_state = min_state;
	tm->boost_true_positive_feedback = boost_true_positive_feedback;
	
	tm->predict = predict;
	tm->update = update;
	
	tm->ta_state = (int8_t ***)malloc(n_clauses * sizeof(int8_t **));  // shape: (n_clauses, n_features, 2)
	if (tm->ta_state == NULL) {
		perror("Memory allocation failed");
		free_tsetlin_machine(tm);
		return NULL;
	}
	for (int i = 0; i < n_clauses; i++) {
		tm->ta_state[i] = (int8_t **)malloc(n_literals * sizeof(int8_t *));
		if (tm->ta_state[i] == NULL) {
			perror("Memory allocation failed");
			free_tsetlin_machine(tm);
			return NULL;
		}
		for (int j = 0; j < n_literals; j++) {
			tm->ta_state[i][j] = (int8_t *)malloc(2 * sizeof(int8_t));
			if (tm->ta_state[i][j] == NULL) {
				perror("Memory allocation failed");
				free_tsetlin_machine(tm);
				return NULL;
			}
		}
	}
	
	tm->weights = (int16_t **)malloc(n_classes * sizeof(int16_t *));  // shape: (n_classes, n_clauses)
	if (tm->weights == NULL) {
		perror("Memory allocation failed");
		free_tsetlin_machine(tm);
		return NULL;
	}
	for (int i = 0; i < n_classes; i++) {
		tm->weights[i] = (int16_t *)malloc(n_clauses * sizeof(int16_t));
		if (tm->weights[i] == NULL) {
			perror("Memory allocation failed");
			free_tsetlin_machine(tm);
			return NULL;
		}
	}
	
	tm->clause_output = (int **)malloc(n_classes * sizeof(int *));  // shape: (n_classes, n_clauses)
	if (tm->clause_output == NULL) {
		perror("Memory allocation failed");
		free_tsetlin_machine(tm);
		return NULL;
	}
	for (int i = 0; i < n_classes; i++) {
		tm->clause_output[i] = (int *)malloc(n_clauses * sizeof(int));
		if (tm->clause_output[i] == NULL) {
			perror("Memory allocation failed");
			free_tsetlin_machine(tm);
			return NULL;
		}
	}
	
	tm->feedback_to_clauses = (int **)malloc(n_classes * sizeof(int *));  // shape: (n_classes, n_clauses)
	if (tm->feedback_to_clauses == NULL) {
		perror("Memory allocation failed");
		free_tsetlin_machine(tm);
		return NULL;
	}
	for (int i = 0; i < n_classes; i++) {
		tm->feedback_to_clauses[i] = (int *)malloc(n_clauses * sizeof(int *));
		if (tm->feedback_to_clauses[i] == NULL) {
			perror("Memory allocation failed");
			free_tsetlin_machine(tm);
			return NULL;
		}
	}

	/* Set up the Tsetlin Machine structure */

	tm_initialize(tm);
	
	return tm;
}


struct TsetlinMachine *load_tsetlin_machine(const char *filename) {
    // I'm assuming correct format, max_state <= 127 && min_state >= -127 && -127 <= max(state) <= 127
    
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
    
    struct TsetlinMachine *tm = create_tsetlin_machine(
        n_classes, threshold, n_literals, n_clauses, max_state, min_state, boost_true_positive_feedback, 1, 0);
    if (!tm) {
        perror("create_tsetlin_machine failed");
        fclose(file);
        return NULL;
    }

    // Allocate and read weights
    size_t weights_size = n_classes * n_clauses * sizeof(int16_t);
    int16_t weights_flat[weights_size];
    fread(weights_flat, weights_size, 1, file);
    
    for (int i = 0; i < n_classes; i++) {				
		for (int j = 0; j < n_clauses; j++) {
			tm->weights[i][j] = weights_flat[(i * n_classes) + j];
		}
	}

    // Allocate and read clauses
    size_t clauses_size = n_clauses * n_literals * 2 * sizeof(int8_t);
    int8_t clauses_flat[clauses_size];
    fread(clauses_flat, clauses_size, 1, file);
    
    for (int i = 0; i < n_clauses; i++) {				
		for (int j = 0; j < n_literals; j++) {
			tm->ta_state[i][j][0] = clauses_flat[((i * n_classes) + j) * 2];
			tm->ta_state[i][j][1] = clauses_flat[(((i * n_classes) + j) * 2) + 1];
		}
	}
	
    fclose(file);
    return tm;
}


void free_tsetlin_machine(struct TsetlinMachine *tm) {
	if (tm != NULL){
		if (tm->ta_state != NULL) {
			for (int i = 0; i < tm->n_clauses; i++) {
				if (tm->ta_state[i] != NULL) {
					for (int j = 0; j < tm->n_literals; j++) {
						if (tm->ta_state[i][j] != NULL) {
							free(tm->ta_state[i][j]);
						}
					}
					free(tm->ta_state[i]);
				}
			}
			free(tm->ta_state);
		}
		
		if (tm->weights != NULL) {
			for (int i = 0; i < tm->n_classes; i++) {
				if (tm->weights[i] != NULL) {
					free(tm->weights[i]);
				}
			}
			free(tm->weights);
		}
		
		if (tm->clause_output != NULL) {
			free(tm->clause_output);
		}
		
		if (tm->feedback_to_clauses != NULL) {
			free(tm->feedback_to_clauses);
		}
		
		free(tm);
	}
	
	return;
}


void tm_initialize(struct TsetlinMachine *tm)
{
    tm->mid_state = (tm->max_state - tm->min_state) / 2;

	for (int j = 0; j < tm->n_clauses; j++) {				
		for (int k = 0; k < tm->n_literals; k++) {
			if (1.0 * rand()/RAND_MAX <= 0.5) {
				tm->ta_state[j][k][0] = tm->mid_state - 1;
				tm->ta_state[j][k][1] = tm->mid_state;
			} else {
				tm->ta_state[j][k][0] = tm->mid_state;
				tm->ta_state[j][k][1] = tm->mid_state - 1; // Deviation, should be random  // What was this comment about
			}
		}
	}
	
	for (int i = 0; i < tm->n_classes; i++) {
		for (int j = 0; j < tm->n_clauses; j++) {
			tm->weights[i][j] = 1;  // TODO: ?
		}
	}
}

/* Translates automata state to action */
static inline int action(int state, int mid_state)
{
		return state >= mid_state;
}

/* Calculate the output of each clause using the actions of each Tsetline Automaton. */
/* Output is stored an internal output array. */

static inline void calculate_clause_output(struct TsetlinMachine *tm, int Xi[], int predict)
{
	int action_include, action_include_negated;
	int all_exclude;

	for (int i = 0; i < tm->n_classes; i++) {
		for (int j = 0; j < tm->n_clauses; j++) {
			tm->clause_output[i][j] = 1;
			all_exclude = 1;
			for (int k = 0; k < tm->n_literals; k++) {
				action_include = action(tm->ta_state[j][k][0], tm->mid_state);
				action_include_negated = action(tm->ta_state[j][k][1], tm->mid_state);
	
				all_exclude = all_exclude && !(action_include == 1 || action_include_negated == 1);
	
				if ((action_include == 1 && Xi[k] == 0) || (action_include_negated == 1 && Xi[k] == 1)) {
					tm->clause_output[i][j] = 0;
					break;
				}
			}
	
			tm->clause_output[i][j] = tm->clause_output[i][j] && !(predict == tm->predict && all_exclude == 1);
		}
	}
}

/* Sum up the votes for each class (this is the multiclass version of the Tsetlin Machine) */
static inline void sum_up_class_votes(struct TsetlinMachine *tm, int *classes_sum)
{
	memset((void *)classes_sum, 0, tm->n_classes*sizeof(int));
	
	for (int i = 0; i < tm->n_classes; i++) {
		for (int j = 0; j < tm->n_clauses; j++) {
			int sign = 1 - 2 * (j & 1);
			classes_sum[i] += tm->clause_output[i][j]*sign * tm->weights[i][j];
		}
		
		classes_sum[i] = (classes_sum[i] > tm->threshold) ? tm->threshold : classes_sum[i];
		classes_sum[i] = (classes_sum[i] < -tm->threshold) ? -tm->threshold : classes_sum[i];
	}
}

/*************************************************/
/*** Type I Feedback (Combats False Negatives) ***/
/*************************************************/

static inline void type_i_feedback(struct TsetlinMachine *tm, int *Xi, int i, int j, float s)
{
	if (tm->clause_output[i][j] == 0) {
		for (int k = 0; k < tm->n_literals; k++) {
			tm->ta_state[j][k][0] -= (tm->ta_state[j][k][0] > tm->min_state) && (1.0*rand()/RAND_MAX <= 1.0/s);
								
			tm->ta_state[j][k][1] -= (tm->ta_state[j][k][1] > tm->min_state) && (1.0*rand()/RAND_MAX <= 1.0/s);
		}
	} else if (tm->clause_output[i][j] == 1) {
		for (int k = 0; k < tm->n_literals; k++) {
			if (Xi[k] == 1) {
				tm->ta_state[j][k][0] += (tm->ta_state[j][k][0] < tm->max_state) && (tm->boost_true_positive_feedback == 1 || 1.0*rand()/RAND_MAX <= (s-1)/s);

				tm->ta_state[j][k][1] -= (tm->ta_state[j][k][1] > tm->min_state) && (1.0*rand()/RAND_MAX <= 1.0/s);
			} else if (Xi[k] == 0) {
				tm->ta_state[j][k][1] += (tm->ta_state[j][k][1] < tm->max_state) && (tm->boost_true_positive_feedback == 1 || 1.0*rand()/RAND_MAX <= (s-1)/s);
				
				tm->ta_state[j][k][0] -= (tm->ta_state[j][k][0] > tm->min_state) && (1.0*rand()/RAND_MAX <= 1.0/s);
			}
		}
	}
}


/**************************************************/
/*** Type II Feedback (Combats False Positives) ***/
/**************************************************/

static inline void type_ii_feedback(struct TsetlinMachine *tm, int *Xi, int i, int j) {
	int action_include;
	int action_include_negated;

	if (tm->clause_output[i][j] == 1) {
		for (int k = 0; k < tm->n_literals; k++) { 
			action_include = action(tm->ta_state[j][k][0], tm->mid_state);
			action_include_negated = action(tm->ta_state[j][k][1], tm->mid_state);

			tm->ta_state[j][k][0] += (action_include == 0 && tm->ta_state[j][k][0] < tm->max_state) && (Xi[k] == 0);
			tm->ta_state[j][k][1] += (action_include_negated == 0 && tm->ta_state[j][k][1] < tm->max_state) && (Xi[k] == 1);
		}
	}
}

/******************************************/
/*** Online Training of Tsetlin Machine ***/
/******************************************/

// The Tsetlin Machine can be trained incrementally, one training example at a time.
// Use this method directly for online and incremental training.

void tm_update(struct TsetlinMachine *tm, int *Xi, int target, float s) {
	/*******************************/
	/*** Calculate Clause Output ***/
	/*******************************/

	calculate_clause_output(tm, Xi, tm->update);

	/***************************/
	/*** Sum up Clause Votes ***/
	/***************************/

	int *classes_sum = (int *)malloc(tm->n_classes * sizeof(int));
	if (classes_sum == NULL) {
		perror("Memory allocation failed");
		free_tsetlin_machine(tm);
		return;
	}
	
	sum_up_class_votes(tm, classes_sum);

	/*************************************/
	/*** Calculate Feedback to Clauses ***/
	/*************************************/

	// Calculate feedback to clauses
	for (int i = 0; i < tm->n_classes; i++) {
		for (int j = 0; j < tm->n_clauses; j++) {
			tm->feedback_to_clauses[i][j] = (2*target-1)*(1 - 2 * (j & 1))*(1.0*rand()/RAND_MAX <= (1.0/(tm->threshold*2))*(tm->threshold + (1 - 2*target)*classes_sum[i]));
		}
	}
	free(classes_sum);
	
	/*********************************/
	/*** Train Individual Automata ***/
	/*********************************/

	for (int i = 0; i < tm->n_classes; i++) {
		for (int j = 0; j < tm->n_clauses; j++) {
			if (tm->feedback_to_clauses[i][j] > 0) {
				type_i_feedback(tm, Xi, i, j, s);
			} else if (tm->feedback_to_clauses[i][j] < 0) {
				type_ii_feedback(tm, Xi, i, j);
			}
		}
	}
}

void tm_score(struct TsetlinMachine *tm, int *Xi, int *result) {
	/*******************************/
	/*** Calculate Clause Output ***/
	/*******************************/

	calculate_clause_output(tm, Xi, tm->predict);

	/***************************/
	/*** Sum up Clause Votes ***/
	/***************************/

	sum_up_class_votes(tm, result);
}

int tm_get_state(struct TsetlinMachine *tm, int clause, int feature, int automaton_type) {
	return tm->ta_state[clause][feature][automaton_type];
}

int tm_get_weight(struct TsetlinMachine *tm, int class_id, int clause) {
	return tm->weights[class_id][clause];
}

