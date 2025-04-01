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

#include <stdio.h>
#include <stdlib.h>

#include "TsetlinMachine.h"

/**************************************/
/*** The Multiclass Tsetlin Machine ***/
/**************************************/

/*** Initialize Tsetlin Machine ***/
struct TsetlinMachine *CreateTsetlinMachine(int threshold, int n_features, int n_clauses, int max_state, int min_state, int boost_true_positive_feedback, int predict, int update)
{
	struct TsetlinMachine *tm = (struct TsetlinMachine *)malloc(sizeof(struct TsetlinMachine));
	if(tm == NULL) {
		perror("Memory allocation failed");
		return NULL;
	}
	
	tm->threshold = threshold;
	tm->n_features = n_features;
	tm->n_clauses = n_clauses;
	tm->max_state = max_state;
	tm->min_state = min_state;
	tm->boost_true_positive_feedback = boost_true_positive_feedback;
	
	tm->predict = predict;
	tm->update = update;
	
	tm->ta_state = (int ***)malloc(n_clauses * sizeof(int **));  // shape: (n_clauses, n_features, 2)
	if (tm->ta_state == NULL) {
		perror("Memory allocation failed");
		free_tsetlin_machine(tm);
		return NULL;
	}
	for (int i = 0; i < n_clauses; i++) {
		tm->ta_state[i] = (int **)malloc(n_features * sizeof(int *));
		if (tm->ta_state[i] == NULL) {
			perror("Memory allocation failed");
			free_tsetlin_machine(tm);
			return NULL;
		}
		for (int j = 0; j < n_features; j++) {
			tm->ta_state[i][j] = (int *)malloc(2 * sizeof(int));
			if (tm->ta_state[i][j] == NULL) {
				perror("Memory allocation failed");
				free_tsetlin_machine(tm);
				return NULL;
			}
		}
	}
	
	tm->clause_output = (int *)malloc(n_clauses * sizeof(int));  // shape: n_clauses
	if (tm->clause_output == NULL) {
		perror("Memory allocation failed");
		free_tsetlin_machine(tm);
		return NULL;
	}
	tm->feedback_to_clauses = (int *)malloc(n_clauses * sizeof(int));  // shape: n_clauses
	if (tm->feedback_to_clauses == NULL) {
		perror("Memory allocation failed");
		free_tsetlin_machine(tm);
		return NULL;
	}

	/* Set up the Tsetlin Machine structure */

	tm_initialize(tm);
	
	return tm;
}


void free_tsetlin_machine(struct TsetlinMachine *tm) {
	if (tm != NULL){
		if (tm->ta_state != NULL) {
			for (int i = 0; i < tm->n_clauses; i++) {
				if (tm->ta_state[i] != NULL) {
					for (int j = 0; j < tm->n_features; j++) {
						if (tm->ta_state[i][j] != NULL) {
							free(tm->ta_state[i][j]);
						}
					}
					free(tm->ta_state[i]);
				}
			}
			free(tm->ta_state);
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
    int mean = (tm->max_state - tm->min_state) / 2;

	for (int j = 0; j < tm->n_clauses; j++) {				
		for (int k = 0; k < tm->n_features; k++) {
			if (1.0 * rand()/RAND_MAX <= 0.5) {
				(*tm).ta_state[j][k][0] = mean - 1;
				(*tm).ta_state[j][k][1] = mean;
			} else {
				(*tm).ta_state[j][k][0] = mean;
				(*tm).ta_state[j][k][1] = mean - 1; // Deviation, should be random  // What is this comment about
			}
		}
	}
}

/* Translates automata state to action */
static inline int action(int state)
{
		return state >= 0;
}

/* Calculate the output of each clause using the actions of each Tsetline Automaton. */
/* Output is stored an internal output array. */

static inline void calculate_clause_output(struct TsetlinMachine *tm, int Xi[], int predict)
{
	int j, k;
	int action_include, action_include_negated;
	int all_exclude;

	for (j = 0; j < tm->n_clauses; j++) {
		(*tm).clause_output[j] = 1;
		all_exclude = 1;
		for (k = 0; k < tm->n_features; k++) {
			action_include = action((*tm).ta_state[j][k][0]);
			action_include_negated = action((*tm).ta_state[j][k][1]);

			all_exclude = all_exclude && !(action_include == 1 || action_include_negated == 1);

			if ((action_include == 1 && Xi[k] == 0) || (action_include_negated == 1 && Xi[k] == 1)) {
				(*tm).clause_output[j] = 0;
				break;
			}
		}

		(*tm).clause_output[j] = (*tm).clause_output[j] && !(predict == tm->predict && all_exclude == 1);
	}
}

/* Sum up the votes for each class (this is the multiclass version of the Tsetlin Machine) */
static inline int sum_up_class_votes(struct TsetlinMachine *tm)
{
	int class_sum = 0;
	for (int j = 0; j < tm->n_clauses; j++) {
		int sign = 1 - 2 * (j & 1);
		class_sum += (*tm).clause_output[j]*sign;
	}
	
	class_sum = (class_sum > tm->threshold) ? tm->threshold : class_sum;
	class_sum = (class_sum < -tm->threshold) ? -tm->threshold : class_sum;

	return class_sum;
}

/* Get the state of a specific automaton, indexed by clause, feature, and automaton type (include/include negated). */
int tm_get_state(struct TsetlinMachine *tm, int clause, int feature, int automaton_type)
{
	return (*tm).ta_state[clause][feature][automaton_type];
}

/*************************************************/
/*** Type I Feedback (Combats False Negatives) ***/
/*************************************************/

static inline void type_i_feedback(struct TsetlinMachine *tm, int *Xi, int j, float s)
{
	if ((*tm).clause_output[j] == 0)	{
		for (int k = 0; k < FEATURES; k++) {
			(*tm).ta_state[j][k][0] -= ((*tm).ta_state[j][k][0] > tm->min_state) && (1.0*rand()/RAND_MAX <= 1.0/s);
								
			(*tm).ta_state[j][k][1] -= ((*tm).ta_state[j][k][1] > tm->min_state) && (1.0*rand()/RAND_MAX <= 1.0/s);
		}
	} else if ((*tm).clause_output[j] == 1) {					
		for (int k = 0; k < FEATURES; k++) {
			if (Xi[k] == 1) {
				(*tm).ta_state[j][k][0] += ((*tm).ta_state[j][k][0] < tm->max_state) && (tm->boost_true_positive_feedback == 1 || 1.0*rand()/RAND_MAX <= (s-1)/s);

				(*tm).ta_state[j][k][1] -= ((*tm).ta_state[j][k][1] > tm->min_state) && (1.0*rand()/RAND_MAX <= 1.0/s);
			} else if (Xi[k] == 0) {
				(*tm).ta_state[j][k][1] += ((*tm).ta_state[j][k][1] < tm->max_state) && (tm->boost_true_positive_feedback == 1 || 1.0*rand()/RAND_MAX <= (s-1)/s);
				
				(*tm).ta_state[j][k][0] -= ((*tm).ta_state[j][k][0] > tm->min_state) && (1.0*rand()/RAND_MAX <= 1.0/s);
			}
		}
	}
}


/**************************************************/
/*** Type II Feedback (Combats False Positives) ***/
/**************************************************/

static inline void type_ii_feedback(struct TsetlinMachine *tm, int *Xi, int j) {
	int action_include;
	int action_include_negated;

	if ((*tm).clause_output[j] == 1) {
		for (int k = 0; k < tm->n_features; k++) { 
			action_include = action((*tm).ta_state[j][k][0]);
			action_include_negated = action((*tm).ta_state[j][k][1]);

			(*tm).ta_state[j][k][0] += (action_include == 0 && (*tm).ta_state[j][k][0] < tm->max_state) && (Xi[k] == 0);
			(*tm).ta_state[j][k][1] += (action_include_negated == 0 && (*tm).ta_state[j][k][1] < tm->max_state) && (Xi[k] == 1);
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

	int class_sum = sum_up_class_votes(tm);

	/*************************************/
	/*** Calculate Feedback to Clauses ***/
	/*************************************/

	// Calculate feedback to clauses
	for (int j = 0; j < tm->n_clauses; j++) {
		(*tm).feedback_to_clauses[j] = (2*target-1)*(1 - 2 * (j & 1))*(1.0*rand()/RAND_MAX <= (1.0/(tm->threshold*2))*(tm->threshold + (1 - 2*target)*class_sum));
	}
	
	/*********************************/
	/*** Train Individual Automata ***/
	/*********************************/

	for (int j = 0; j < tm->n_clauses; j++) {
		if ((*tm).feedback_to_clauses[j] > 0) {
			type_i_feedback(tm, Xi, j, s);
		} else if ((*tm).feedback_to_clauses[j] < 0) {
			type_ii_feedback(tm, Xi, j);
		}
	}
}

int tm_score(struct TsetlinMachine *tm, int *Xi) {
	/*******************************/
	/*** Calculate Clause Output ***/
	/*******************************/

	calculate_clause_output(tm, Xi, tm->predict);

	/***************************/
	/*** Sum up Clause Votes ***/
	/***************************/

	return sum_up_class_votes(tm);
}


