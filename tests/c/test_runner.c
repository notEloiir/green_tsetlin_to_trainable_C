/*
 * test_runner.c
 *
 *  Created on: 11 May 2025
 *      Author: Elouie
 */


#include "tsetlin_machine.h"
#include "unity/unity.h"
#include "stdlib.h"


void setUp(void) {
    srand(42);
}
void tearDown(void) {}


extern void test_tsetlin_machine_run_all(void);
extern void test_linked_list_run_all(void);


int main(void) {
    UNITY_BEGIN();

    test_tsetlin_machine_run_all();
    test_linked_list_run_all();

    return UNITY_END();
}

