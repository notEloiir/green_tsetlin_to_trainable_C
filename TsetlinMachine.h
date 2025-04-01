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

#define THRESHOLD 15
#define FEATURES 12
#define CLAUSES 10
#define NUMBER_OF_STATES 100
#define BOOST_TRUE_POSITIVE_FEEDBACK 0

#define PREDICT 1
#define UPDATE 0

struct TsetlinMachine {
    int threshold;
    int n_features;
    int n_clauses;
    int max_state, min_state;
    int boost_true_positive_feedback;

    int predict;
    int update;

	int ***ta_state;  // shape: (n_clauses, n_features, 2)
	int *clause_output;  // shape: n_clauses
	int *feedback_to_clauses;  // shape: n_clauses
};

struct TsetlinMachine *CreateTsetlinMachine(
    int threshold, int n_features, int n_clauses, int max_state, int min_state, int boost_true_positive_feedback, int predict, int update
);

void free_tsetlin_machine(struct TsetlinMachine *tm);

void tm_initialize(struct TsetlinMachine *tm);

void tm_update(struct TsetlinMachine *tm, int *Xi, int target, float s);

int tm_score(struct TsetlinMachine *tm, int *Xi);

int tm_get_state(struct TsetlinMachine *tm, int clause, int feature, int automaton_type);

