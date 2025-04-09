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

struct TsetlinMachine {
	int num_classes;
    int threshold;
    int num_literals;
    int num_clauses;
    int max_state, min_state;
    int boost_true_positive_feedback;

    int predict;
    int update;

	int mid_state;
	int8_t *ta_state;  // shape: flat (num_clauses, num_literals, 2)
	int16_t *weights;  // shape: flat (num_classes, num_clauses)
	int *clause_output;  // shape: (num_clauses)
	int *clause_feedback;  // shape: flat (num_classes, num_clauses)
};

// Input shape: (num_literals)
// Output shape: (num_classes)

// Create a Tsetlin Machine. Number of classes corresponds to number of bits in the TM output.
struct TsetlinMachine *create_tsetlin_machine(
    int num_classes, int threshold, int num_literals, int num_clauses, int max_state, int min_state, int boost_true_positive_feedback, int predict, int update
);

struct TsetlinMachine *load_tsetlin_machine(const char *filename);

// Deallocate all memory.
void free_tsetlin_machine(struct TsetlinMachine *tm);

// Train on a single data point.
void tm_update(struct TsetlinMachine *tm, uint8_t *Xi, int target, float s);

// Inference on a single data point. Writes to the result array of size (num_classes), elements in range [-threshold, threshold].
void tm_score(struct TsetlinMachine *tm, uint8_t *Xi, int *result);

int tm_get_state(struct TsetlinMachine *tm, int clause_id, int literal_id, int automaton_type);

int tm_get_weight(struct TsetlinMachine *tm, int class_id, int clause_id);

