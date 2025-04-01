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

#include "TsetlinMachine.h"


struct MultiClassTsetlinMachine { 
	int n_classes;
	
	struct TsetlinMachine **tsetlin_machines;  // shape: n_classes
};

struct MultiClassTsetlinMachine *CreateMultiClassTsetlinMachine(
	int n_classes, int threshold, int n_features, int n_clauses, int max_state, int min_state, int boost_true_positive_feedback, int predict, int update
);

void free_mc_tsetlin_machine(struct MultiClassTsetlinMachine *tm);

void mc_tm_initialize(struct MultiClassTsetlinMachine *mc_tm);

// X shape: *, n_features
// y shape: *
float mc_tm_evaluate(struct MultiClassTsetlinMachine *mc_tm, int **X, int *y, int number_of_examples);

void mc_tm_fit(struct MultiClassTsetlinMachine *mc_tm, int **X, int *y, int number_of_examples, int epochs, float s);
