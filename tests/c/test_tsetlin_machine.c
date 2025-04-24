#include "../../src/c/include/tsetlin_machine.h"
#include "../../src/c/include/tm_output_activation.h"
#include "unity/unity.h"
#include "stdlib.h"


void setUp(void) {
    srand(42);
}
void tearDown(void) {}

void basic_inference(void) {
    struct TsetlinMachine *tm = tm_create(1, 100, 3, 1, 127, -127, 0, 1, sizeof(uint8_t), 10.f);
    // One clause that "activates" on literal values: 10x where x means any
    tm->ta_state[0] = 1; tm->ta_state[1] = -1;
    tm->ta_state[2] = -1; tm->ta_state[3] = 1;
    tm->ta_state[4] = -1; tm->ta_state[5] = -1;
    // And its vote has weight 1
    tm->weights[0] = 1;
    // Set output_activation to binary vector, instead of default class argmax
    tm_set_output_activation(tm, oa_bin_vector);
    tm_set_calculate_feedback(tm, feedback_bin_vector);  // Not used here, real usage example
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
    tm_set_output_activation(tm, oa_bin_vector);
    tm_set_calculate_feedback(tm, feedback_bin_vector);  // Not used here, real usage example
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

int main(void) {
    UNITY_BEGIN();

    RUN_TEST(basic_inference);
    RUN_TEST(basic_training);

    return UNITY_END();
}
