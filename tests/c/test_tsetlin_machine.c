#include "../../src/c/include/tsetlin_machine.h"
#include "unity/unity.h"
#include "stdlib.h"

#include "../../src/c/src/tsetlin_machine.c"
#include "../../src/c/src/utility.c"


void basic_inference(void) {
    struct TsetlinMachine *tm = tm_create(1, 100, 3, 1, 127, -127, 0, 1, sizeof(uint8_t), 10.f);
    // One clause that "activates" on literal values: 10x where x means any
    tm->ta_state[0] = 1; tm->ta_state[1] = -1;
    tm->ta_state[2] = -1; tm->ta_state[3] = 1;
    tm->ta_state[4] = -1; tm->ta_state[5] = -1;
    // And its vote has weight 1
    tm->weights[0] = 1;
    // Set output_activation to binary vector, instead of default class argmax
    tm_set_output_activation(tm, tm_oa_bin_vector);
    tm_set_calculate_feedback(tm, tm_feedback_bin_vector);  // Not used here, real usage example
    // Allocate memory for X and y_pred
    uint8_t *X = malloc(3 * sizeof(uint8_t));
    uint8_t *y_pred = malloc(1 * sizeof(uint8_t));

    // Input 100 should result in output 1
    X[0] = 1;
    X[1] = 0;
    X[2] = 0;
    tm_predict(tm, X, y_pred, 1);
    TEST_ASSERT_EQUAL_INT(1, y_pred[0]);

    // Input 110 should result in output 0
    X[0] = 1;
    X[1] = 1;
    X[2] = 0;
    tm_predict(tm, X, y_pred, 1);
    TEST_ASSERT_EQUAL_INT(0, y_pred[0]);

    tm_free(tm);
    free(X);
    free(y_pred);
}

void basic_training(void) {
    struct TsetlinMachine *tm = tm_create(1, 100, 3, 1, 127, -127, 0, 1, sizeof(uint8_t), 10.f);
    // One clause that "activates" on literal values: 10x where x means any
    tm->ta_state[0] = 1; tm->ta_state[1] = -1;
    tm->ta_state[2] = -1; tm->ta_state[3] = 1;
    tm->ta_state[4] = -1; tm->ta_state[5] = -1;
    // And its vote has weight 1
    tm->weights[0] = 1;
    // Set output_activation to binary vector, instead of default class argmax
    tm_set_output_activation(tm, tm_oa_bin_vector);
    tm_set_calculate_feedback(tm, tm_feedback_bin_vector);  // Not used here, real usage example
    // Allocate memory for X and y_pred
    uint8_t *X = malloc(3 * sizeof(uint8_t));
    uint8_t *y_pred = malloc(1 * sizeof(uint8_t));

    // Training input: 101 should output 1 before training, 0 after training
    X[0] = 1;
    X[1] = 0;
    X[2] = 1;
    tm_predict(tm, X, y_pred, 1);
    TEST_ASSERT_EQUAL_INT(1, y_pred[0]);

    uint8_t *y = malloc(1 * sizeof(uint8_t));
    y[0] = 0;
    tm_train(tm, X, y, 1, 1, 10);  // 1 datapoint, 10 epochs

    tm_predict(tm, X, y_pred, 1);
    TEST_ASSERT_EQUAL_INT(0, y_pred[0]);

    tm_free(tm);
    free(X);
    free(y_pred);
    free(y);
}

void test_calculate_clause_output(void) {
    struct TsetlinMachine *tm = tm_create(1, 50, 2, 2, 127, -127, 0, 1, sizeof(uint8_t), 10.f);
    tm->ta_state[0] = 100;
    tm->ta_state[1] = -100;
    tm->ta_state[2] = 100;
    tm->ta_state[3] = -100;
    tm->ta_state[4] = -100;
    tm->ta_state[5] = 100;
    tm->ta_state[6] = -100;
    tm->ta_state[7] = 100;

    uint8_t X[] = {1, 1};

    calculate_clause_output(tm, X);
    TEST_ASSERT_EQUAL_INT(1, tm->clause_output[0]);
    TEST_ASSERT_EQUAL_INT(0, tm->clause_output[1]);

    tm_free(tm);
}

void test_sum_votes(void) {
    struct TsetlinMachine *tm = tm_create(2, 100, 2, 2, 127, -127, 0, 1, sizeof(uint8_t), 10.f);

    tm->clause_output[0] = 1;
    tm->clause_output[1] = 0;

    tm->weights[0] = 5;
    tm->weights[1] = 0;
    tm->weights[3] = 1;
    tm->weights[4] = 10;

    sum_votes(tm);

    TEST_ASSERT_EQUAL_INT(5, tm->votes[0]);
    TEST_ASSERT_EQUAL_INT(0, tm->votes[1]);

    tm_free(tm);
}


void test_type_1a_feedback(void) {
    struct TsetlinMachine *tm = tm_create(1, 100, 3, 1, 127, -127, 1, 1, sizeof(uint8_t), 10.f);
    tm->ta_state[0] = 1; tm->ta_state[1] = -1;
    tm->ta_state[2] = -1; tm->ta_state[3] = 1;
    tm->ta_state[4] = -1; tm->ta_state[5] = -1;

    tm->weights[0] = 1;
    tm->feedback[0] = 1;

    uint8_t X[] = {1, 0, 0};

    type_1a_feedback(tm, X);

    TEST_ASSERT_EQUAL_INT(2, tm->weights[0]);

    TEST_ASSERT_EQUAL_INT(2, tm->ta_state[0]);
    TEST_ASSERT_EQUAL_INT(2, tm->ta_state[3]);
    TEST_ASSERT_EQUAL_INT(0, tm->ta_state[5]);

    TEST_ASSERT_EQUAL_INT(-1, tm->ta_state[1]);
    TEST_ASSERT_EQUAL_INT(-1, tm->ta_state[2]);
    TEST_ASSERT_EQUAL_INT(-1, tm->ta_state[4]);

    tm_free(tm);
}

void test_type_1b_feedback(void) {
    struct TsetlinMachine *tm = tm_create(1, 100, 3, 1, 127, -127, 1, 1, sizeof(uint8_t), 1.f);
    tm->feedback[1] = 1;

    tm->ta_state[0] = 1; tm->ta_state[1] = -1;
    tm->ta_state[2] = -1; tm->ta_state[3] = 1;
    tm->ta_state[4] = -1; tm->ta_state[5] = -1;

    type_1b_feedback(tm);

    TEST_ASSERT_EQUAL_INT(0, tm->ta_state[0]);
    TEST_ASSERT_EQUAL_INT(-2, tm->ta_state[1]);
    TEST_ASSERT_EQUAL_INT(-2, tm->ta_state[2]);
    TEST_ASSERT_EQUAL_INT(0, tm->ta_state[3]);
    TEST_ASSERT_EQUAL_INT(-2, tm->ta_state[4]);
    TEST_ASSERT_EQUAL_INT(-2, tm->ta_state[5]);

    tm_free(tm);
}

void test_type_2_feedback(void) {
    struct TsetlinMachine *tm = tm_create(1, 100, 3, 1, 127, -127, 1, 1, sizeof(uint8_t), 1.f);
    tm->feedback[2] = 1;

    tm->ta_state[0] = 1; tm->ta_state[1] = -1;
    tm->ta_state[2] = -1; tm->ta_state[3] = 1;
    tm->ta_state[4] = -1; tm->ta_state[5] = -1;

    uint8_t X[] = {1, 0, 1};

    type_2_feedback(tm, X);

    TEST_ASSERT_EQUAL_INT(1, tm->ta_state[0]);
    TEST_ASSERT_EQUAL_INT(1, tm->ta_state[3]);
    TEST_ASSERT_EQUAL_INT(-1, tm->ta_state[4]);

    TEST_ASSERT_EQUAL_INT(0, tm->ta_state[1]);
    TEST_ASSERT_EQUAL_INT(0, tm->ta_state[2]);
    TEST_ASSERT_EQUAL_INT(0, tm->ta_state[5]);

    tm_free(tm);
}

void test_tsetlin_machine_run_all(void) {
    RUN_TEST(basic_inference);
    RUN_TEST(basic_training);
    RUN_TEST(test_calculate_clause_output);
    RUN_TEST(test_sum_votes);
    RUN_TEST(test_type_1a_feedback);
    RUN_TEST(test_type_1b_feedback);
    RUN_TEST(test_type_2_feedback);
}
