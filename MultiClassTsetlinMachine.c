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

This code implements a multiclass version of the Tsetlin Machine from paper arXiv:1804.01508
https://arxiv.org/abs/1804.01508

*/

#include <stdio.h>
#include <stdlib.h>

#include "MultiClassTsetlinMachine.h"

/**************************************/
/*** The Multiclass Tsetlin Machine ***/
/**************************************/

/*** Initialize Tsetlin Machine ***/
struct MultiClassTsetlinMachine *CreateMultiClassTsetlinMachine(
	int n_classes, int threshold, int n_literals, int n_clauses, int max_state, int min_state, int boost_true_positive_feedback, int predict, int update
)
{
	struct MultiClassTsetlinMachine *mc_tm = (struct MultiClassTsetlinMachine *)malloc(sizeof(struct MultiClassTsetlinMachine));
	if(mc_tm == NULL) {
		perror("Memory allocation failed");
		return NULL;
	}
	
	mc_tm->n_classes = n_classes;
	
	mc_tm->tsetlin_machines = (struct TsetlinMachine **)malloc(n_classes * sizeof(struct TsetlinMachine *));
	if(mc_tm->tsetlin_machines == NULL) {
		perror("Memory allocation failed");
		return NULL;
	}

	for (int i = 0; i < mc_tm->n_classes; i++) {
		mc_tm->tsetlin_machines[i] = CreateTsetlinMachine(threshold, n_literals, n_clauses, max_state, min_state, boost_true_positive_feedback, predict, update);
		if (mc_tm == NULL) {
			perror("CreateTsetlinMachine failed");
			free_mc_tsetlin_machine(mc_tm);
			return NULL;
		}
	}
	return mc_tm;
}

void free_mc_tsetlin_machine(struct MultiClassTsetlinMachine *mc_tm) {
	if (mc_tm != NULL) {
		if (mc_tm->tsetlin_machines != NULL) {
			for (int i = 0; i < mc_tm->n_classes; i++) {
				if(mc_tm->tsetlin_machines[i] != NULL) {
					free_tsetlin_machine(mc_tm->tsetlin_machines[i]);
				}
			}
			free(mc_tm->tsetlin_machines);
		}
		free(mc_tm);
	}
}

void mc_tm_initialize(struct MultiClassTsetlinMachine *mc_tm)
{
	for (int i = 0; i < mc_tm->n_classes; i++) {
		tm_initialize(mc_tm->tsetlin_machines[i]);
	}
}

/********************************************/
/*** Evaluate the Trained Tsetlin Machine ***/
/********************************************/

float mc_tm_evaluate(struct MultiClassTsetlinMachine *mc_tm, int **X, int *y, int number_of_examples)
{
	int errors;
	int max_class;
	int max_class_sum;

	errors = 0;
	for (int l = 0; l < number_of_examples; l++) {
		/******************************************/
		/*** Identify Class with Largest Output ***/
		/******************************************/

		max_class_sum = tm_score(mc_tm->tsetlin_machines[0], X[l]);
		max_class = 0;
		for (int i = 1; i < mc_tm->n_classes; i++) {	
			int class_sum = tm_score(mc_tm->tsetlin_machines[i], X[l]);
			if (max_class_sum < class_sum) {
				max_class_sum = class_sum;
				max_class = i;
			}
		}

		if (max_class != y[l]) {
			errors += 1;
		}
	}
	
	return 1.0 - 1.0 * errors / number_of_examples;
}

/******************************************/
/*** Online Training of Tsetlin Machine ***/
/******************************************/

// The Tsetlin Machine can be trained incrementally, one training example at a time.
// Use this method directly for online and incremental training.

void mc_tm_update(struct MultiClassTsetlinMachine *mc_tm, int *Xi, int target_class, float s)
{
	tm_update(mc_tm->tsetlin_machines[target_class], Xi, 1, s);

	// Randomly pick one of the other classes, for pairwise learning of class output 
	unsigned int negative_target_class = (unsigned int)mc_tm->n_classes * 1.0*rand()/((unsigned int)RAND_MAX+1);
	while (negative_target_class == target_class) {
		negative_target_class = (unsigned int)mc_tm->n_classes * 1.0*rand()/((unsigned int)RAND_MAX+1);
	}

	tm_update(mc_tm->tsetlin_machines[negative_target_class], Xi, 0, s);
}

/**********************************************/
/*** Batch Mode Training of Tsetlin Machine ***/
/**********************************************/

void mc_tm_fit(struct MultiClassTsetlinMachine *mc_tm, int **X, int *y, int number_of_examples, int epochs, float s)
{
	for (int epoch = 0; epoch < epochs; epoch++) {
		// Add shuffling here...		
		for (int i = 0; i < number_of_examples; i++) {
			mc_tm_update(mc_tm, X[i], y[i], s);
		}
	}
}

