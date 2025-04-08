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
struct TsetlinMachine *create_tsetlin_machine(int num_classes, int threshold, int num_literals, int num_clauses, int max_state, int min_state, int boost_true_positive_feedback, int predict, int update)
{
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
	
	tm->predict = predict;
	tm->update = update;
	
	tm->ta_state = (int8_t ***)malloc(num_clauses * sizeof(int8_t **));  // shape: (num_clauses, n_features, 2)
	if (tm->ta_state == NULL) {
		perror("Memory allocation failed");
		free_tsetlin_machine(tm);
		return NULL;
	}
	for (int clause_id = 0; clause_id < num_clauses; clause_id++) {
		tm->ta_state[clause_id] = (int8_t **)malloc(num_literals * sizeof(int8_t *));
		if (tm->ta_state[clause_id] == NULL) {
			perror("Memory allocation failed");
			free_tsetlin_machine(tm);
			return NULL;
		}
		for (int literal_id = 0; literal_id < num_literals; literal_id++) {
			tm->ta_state[clause_id][literal_id] = (int8_t *)malloc(2 * sizeof(int8_t));
			if (tm->ta_state[clause_id][literal_id] == NULL) {
				perror("Memory allocation failed");
				free_tsetlin_machine(tm);
				return NULL;
			}
		}
	}
	
	tm->weights = (int16_t **)malloc(num_classes * sizeof(int16_t *));  // shape: (num_classes, num_clauses)
	if (tm->weights == NULL) {
		perror("Memory allocation failed");
		free_tsetlin_machine(tm);
		return NULL;
	}
	for (int class_id = 0; class_id < num_classes; class_id++) {
		tm->weights[class_id] = (int16_t *)malloc(num_clauses * sizeof(int16_t));
		if (tm->weights[class_id] == NULL) {
			perror("Memory allocation failed");
			free_tsetlin_machine(tm);
			return NULL;
		}
	}
	
	tm->clause_output = (int *)malloc(num_clauses * sizeof(int));  // shape: (num_clauses)
	if (tm->clause_output == NULL) {
		perror("Memory allocation failed");
		free_tsetlin_machine(tm);
		return NULL;
	}
	
	tm->clause_feedback = (int **)malloc(num_classes * sizeof(int *));  // shape: (num_classes, num_clauses)
	if (tm->clause_feedback == NULL) {
		perror("Memory allocation failed");
		free_tsetlin_machine(tm);
		return NULL;
	}
	for (int class_id = 0; class_id < num_classes; class_id++) {
		tm->clause_feedback[class_id] = (int *)malloc(num_clauses * sizeof(int *));
		if (tm->clause_feedback[class_id] == NULL) {
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
    int num_literals, num_clauses, num_classes;
    int max_state, min_state;
    int boost_true_positive_feedback;

    // Read metadata
    fread(&threshold, sizeof(int), 1, file);
    fread(&num_literals, sizeof(int), 1, file);
    fread(&num_clauses, sizeof(int), 1, file);
    fread(&num_classes, sizeof(int), 1, file);
    fread(&max_state, sizeof(int), 1, file);
    fread(&min_state, sizeof(int), 1, file);
    fread(&boost_true_positive_feedback, sizeof(int), 1, file); 
    
    struct TsetlinMachine *tm = create_tsetlin_machine(
        num_classes, threshold, num_literals, num_clauses, max_state, min_state, boost_true_positive_feedback, 1, 0);
    if (!tm) {
        perror("create_tsetlin_machine failed");
        fclose(file);
        return NULL;
    }

    // Allocate and read weights
    size_t weights_size = num_classes * num_clauses * sizeof(int16_t);
    int16_t weights_flat[weights_size];
    fread(weights_flat, weights_size, 1, file);
    
    for (int class_id = 0; class_id < num_classes; class_id++) {				
		for (int clause_id = 0; clause_id < num_clauses; clause_id++) {
			tm->weights[class_id][clause_id] = weights_flat[(class_id * num_clauses) + clause_id];
		}
	}

    // Allocate and read clauses
    size_t clauses_size = num_clauses * num_literals * 2 * sizeof(int8_t);
    int8_t clauses_flat[clauses_size];
    fread(clauses_flat, clauses_size, 1, file);
    
    for (int clause_id = 0; clause_id < num_clauses; clause_id++) {				
		for (int literal_id = 0; literal_id < num_literals; literal_id++) {
			tm->ta_state[clause_id][literal_id][0] = clauses_flat[((clause_id * num_literals) + literal_id) * 2];
			tm->ta_state[clause_id][literal_id][1] = clauses_flat[(((clause_id * num_literals) + literal_id) * 2) + 1];
		}
	}
	
    fclose(file);
    return tm;
}


void free_tsetlin_machine(struct TsetlinMachine *tm) {
	if (tm != NULL){
		if (tm->ta_state != NULL) {
			for (int clause_id = 0; clause_id < tm->num_clauses; clause_id++) {
				if (tm->ta_state[clause_id] != NULL) {
					for (int literal_id = 0; literal_id < tm->num_literals; literal_id++) {
						if (tm->ta_state[clause_id][literal_id] != NULL) {
							free(tm->ta_state[clause_id][literal_id]);
						}
					}
					free(tm->ta_state[clause_id]);
				}
			}
			free(tm->ta_state);
		}
		
		if (tm->weights != NULL) {
			for (int class_id = 0; class_id < tm->num_classes; class_id++) {
				if (tm->weights[class_id] != NULL) {
					free(tm->weights[class_id]);
				}
			}
			free(tm->weights);
		}
		
		if (tm->clause_output != NULL) {
			free(tm->clause_output);
		}
		
		if (tm->clause_feedback != NULL) {
			free(tm->clause_feedback);
		}
		
		free(tm);
	}
	
	return;
}


void tm_initialize(struct TsetlinMachine *tm)
{
    tm->mid_state = (tm->max_state + tm->min_state) / 2;

	for (int clause_id = 0; clause_id < tm->num_clauses; clause_id++) {				
		for (int literal_id = 0; literal_id < tm->num_literals; literal_id++) {
			if (1.0 * rand()/RAND_MAX <= 0.5) {
				tm->ta_state[clause_id][literal_id][0] = tm->mid_state - 1;
				tm->ta_state[clause_id][literal_id][1] = tm->mid_state;
			} else {
				tm->ta_state[clause_id][literal_id][0] = tm->mid_state;
				tm->ta_state[clause_id][literal_id][1] = tm->mid_state - 1; // Deviation, should be random  // What was this comment about
			}
		}
	}
	
	for (int class_id = 0; class_id < tm->num_classes; class_id++) {
		for (int clause_id = 0; clause_id < tm->num_clauses; clause_id++) {
			tm->weights[class_id][clause_id] = 1;  // TODO: ?
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

static inline void calculate_clause_output(struct TsetlinMachine *tm, uint8_t *Xi)
{
	int action_include, action_include_negated;
	int empty_clause;

	for (int clause_id = 0; clause_id < tm->num_clauses; clause_id++) {
		tm->clause_output[clause_id] = 1;
		empty_clause = 1;
		for (int literal_id = 0; literal_id < tm->num_literals; literal_id++) {
			action_include = action(tm->ta_state[clause_id][literal_id][0], tm->mid_state);
			action_include_negated = action(tm->ta_state[clause_id][literal_id][1], tm->mid_state);
			
			empty_clause = (empty_clause && (action_include || action_include_negated));

			if ((action_include == 1 && Xi[literal_id] == 0) || (action_include_negated == 1 && Xi[literal_id] == 1) || empty_clause) {
				tm->clause_output[clause_id] = 0;
				break;
			}
		}
	}
}

/* Sum up the votes for each class (this is the multiclass version of the Tsetlin Machine) */
static inline void sum_up_class_votes(struct TsetlinMachine *tm, int *classes_sum)
{
	memset((void *)classes_sum, 0, tm->num_classes*sizeof(int));
	
	for (int clause_id = 0; clause_id < tm->num_clauses; clause_id++) {
		if (tm->clause_output[clause_id] == 0) {
			continue;
		}
		
		for (int class_id = 0; class_id < tm->num_classes; class_id++) {
			classes_sum[class_id] += tm->weights[class_id][clause_id];
		}
	}
	
	for (int class_id = 0; class_id < tm->num_classes; class_id++) {
		classes_sum[class_id] = (classes_sum[class_id] > tm->threshold) ? tm->threshold : classes_sum[class_id];
		classes_sum[class_id] = (classes_sum[class_id] < -tm->threshold) ? -tm->threshold : classes_sum[class_id];
	}
}

/*************************************************/
/*** Type I Feedback (Combats False Negatives) ***/
/*************************************************/

static inline void type_i_feedback(struct TsetlinMachine *tm, uint8_t *Xi, int class_id, int clause_id, float s)
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

static inline void type_ii_feedback(struct TsetlinMachine *tm, uint8_t *Xi, int class_id, int clause_id) {
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

/******************************************/
/*** Online Training of Tsetlin Machine ***/
/******************************************/

// The Tsetlin Machine can be trained incrementally, one training example at a time.
// Use this method directly for online and incremental training.

void tm_update(struct TsetlinMachine *tm, uint8_t *Xi, int target, float s) {
	/*******************************/
	/*** Calculate Clause Output ***/
	/*******************************/

	calculate_clause_output(tm, Xi);

	/***************************/
	/*** Sum up Clause Votes ***/
	/***************************/

	int *classes_sum = (int *)malloc(tm->num_classes * sizeof(int));
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
	for (int class_id = 0; class_id < tm->num_classes; class_id++) {
		for (int clause_id = 0; clause_id < tm->num_clauses; clause_id++) {
			tm->clause_feedback[class_id][clause_id] = (2*target-1)*(1 - 2 * (clause_id & 1))*(1.0*rand()/RAND_MAX <= (1.0/(tm->threshold*2))*(tm->threshold + (1 - 2*target)*classes_sum[class_id]));
		}
	}
	free(classes_sum);
	
	/*********************************/
	/*** Train Individual Automata ***/
	/*********************************/

	for (int class_id = 0; class_id < tm->num_classes; class_id++) {
		for (int clause_id = 0; clause_id < tm->num_clauses; clause_id++) {
			if (tm->clause_feedback[class_id][clause_id] > 0) {
				type_i_feedback(tm, Xi, class_id, clause_id, s);
			} else if (tm->clause_feedback[class_id][clause_id] < 0) {
				type_ii_feedback(tm, Xi, class_id, clause_id);
			}
		}
	}
}

void tm_score(struct TsetlinMachine *tm, uint8_t *Xi, int *result) {
	/*******************************/
	/*** Calculate Clause Output ***/
	/*******************************/

	calculate_clause_output(tm, Xi);

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

