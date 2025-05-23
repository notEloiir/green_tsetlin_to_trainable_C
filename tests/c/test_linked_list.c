#include "sparse_tsetlin_machine.h"
#include "unity/unity.h"
#include "stdlib.h"


#include "../../src/c/src/sparse_tsetlin_machine.c"

void insert_nodes(void) {
	struct TAStateNode *head = NULL;
	struct TAStateNode **head_ptr = &head;

	ta_state_insert(head_ptr, NULL, 2, 4, NULL);
	TEST_ASSERT_NOT_EQUAL(NULL, head);
	TEST_ASSERT_EQUAL(2, head->ta_id);
	TEST_ASSERT_EQUAL(4, head->ta_state);
	TEST_ASSERT_EQUAL(NULL, head->next);
//	printf("Appended at the start.  IDs: 2  States: 4\n");

	ta_state_insert(head_ptr, NULL, 0, 5, NULL);
	TEST_ASSERT_NOT_EQUAL(NULL, head);
	TEST_ASSERT_EQUAL(0, head->ta_id);
	TEST_ASSERT_EQUAL(5, head->ta_state);
	TEST_ASSERT_NOT_EQUAL(NULL, head->next);
	struct TAStateNode *second = head->next;
	TEST_ASSERT_NOT_EQUAL(NULL, second);
	TEST_ASSERT_EQUAL(2, second->ta_id);
	TEST_ASSERT_EQUAL(4, second->ta_state);
	TEST_ASSERT_EQUAL(NULL, second->next);
//	printf("Appended at the start.  IDs: 02  States: 54\n");

	ta_state_insert(head_ptr, head->next, 3, 6, NULL);
	TEST_ASSERT_NOT_EQUAL(NULL, head);
	TEST_ASSERT_EQUAL(0, head->ta_id);
	TEST_ASSERT_EQUAL(5, head->ta_state);
	TEST_ASSERT_NOT_EQUAL(NULL, head->next);
	TEST_ASSERT_NOT_EQUAL(NULL, second);
	TEST_ASSERT_EQUAL(2, second->ta_id);
	TEST_ASSERT_EQUAL(4, second->ta_state);
	TEST_ASSERT_NOT_EQUAL(NULL, second->next);
	struct TAStateNode *third = second->next;
	TEST_ASSERT_NOT_EQUAL(NULL, third);
	TEST_ASSERT_EQUAL(3, third->ta_id);
	TEST_ASSERT_EQUAL(6, third->ta_state);
	TEST_ASSERT_EQUAL(NULL, third->next);
//	printf("Appended at the end.  IDs: 023  States: 546\n");

	ta_state_insert(head_ptr, head, 1, 7, NULL);
	TEST_ASSERT_NOT_EQUAL(NULL, head);
	TEST_ASSERT_EQUAL(0, head->ta_id);
	TEST_ASSERT_EQUAL(5, head->ta_state);
	TEST_ASSERT_NOT_EQUAL(NULL, head->next);
	TEST_ASSERT_NOT_EQUAL(NULL, second);
	TEST_ASSERT_EQUAL(2, second->ta_id);
	TEST_ASSERT_EQUAL(4, second->ta_state);
	TEST_ASSERT_NOT_EQUAL(NULL, second->next);
	TEST_ASSERT_NOT_EQUAL(NULL, third);
	TEST_ASSERT_EQUAL(3, third->ta_id);
	TEST_ASSERT_EQUAL(6, third->ta_state);
	TEST_ASSERT_EQUAL(NULL, third->next);
	struct TAStateNode *first = head->next;
	TEST_ASSERT_NOT_EQUAL(NULL, first);
	TEST_ASSERT_EQUAL(1, first->ta_id);
	TEST_ASSERT_EQUAL(7, first->ta_state);
	TEST_ASSERT_NOT_EQUAL(NULL, first->next);
//	printf("Appended in the middle.  IDs: 0123  States: 5746\n");
}

void remove_nodes(void) {
	struct TAStateNode *head = NULL;
	struct TAStateNode **head_ptr = &head;
	ta_state_insert(head_ptr, NULL, 2, 4, NULL);
	ta_state_insert(head_ptr, NULL, 0, 5, NULL);
	ta_state_insert(head_ptr, head->next, 3, 6, NULL);
	ta_state_insert(head_ptr, head, 1, 7, NULL);
	struct TAStateNode *first = head->next;
	struct TAStateNode *second = first->next;
	struct TAStateNode *third = second->next;
//	printf("Start.  IDs: 0123  States: 5746\n");

	ta_state_remove(head_ptr, head, NULL);
	TEST_ASSERT_NOT_EQUAL(NULL, head);
	TEST_ASSERT_EQUAL(0, head->ta_id);
	TEST_ASSERT_EQUAL(5, head->ta_state);
	TEST_ASSERT_NOT_EQUAL(NULL, head->next);
	TEST_ASSERT_NOT_EQUAL(NULL, second);
	TEST_ASSERT_EQUAL(2, second->ta_id);
	TEST_ASSERT_EQUAL(4, second->ta_state);
	TEST_ASSERT_NOT_EQUAL(NULL, second->next);
	TEST_ASSERT_NOT_EQUAL(NULL, third);
	TEST_ASSERT_EQUAL(3, third->ta_id);
	TEST_ASSERT_EQUAL(6, third->ta_state);
	TEST_ASSERT_EQUAL(NULL, third->next);
//	printf("Removed in the middle.  IDs: 023  States: 546\n");

	ta_state_remove(head_ptr, head->next, NULL);
	TEST_ASSERT_NOT_EQUAL(NULL, head);
	TEST_ASSERT_EQUAL(0, head->ta_id);
	TEST_ASSERT_EQUAL(5, head->ta_state);
	TEST_ASSERT_NOT_EQUAL(NULL, head->next);
	TEST_ASSERT_NOT_EQUAL(NULL, second);
	TEST_ASSERT_EQUAL(2, second->ta_id);
	TEST_ASSERT_EQUAL(4, second->ta_state);
	TEST_ASSERT_EQUAL(NULL, second->next);
//	printf("Appended at the end.  IDs: 02  States: 54\n");

	ta_state_remove(head_ptr, NULL, NULL);
	TEST_ASSERT_NOT_EQUAL(NULL, head);
	TEST_ASSERT_EQUAL(2, head->ta_id);
	TEST_ASSERT_EQUAL(4, head->ta_state);
	TEST_ASSERT_EQUAL(NULL, head->next);
//	printf("Appended at the start.  IDs: 2  States: 4\n");

	ta_state_remove(head_ptr, NULL, NULL);
	TEST_ASSERT_EQUAL(NULL, head);
//	printf("Appended at the start.  IDs: -  States: -\n");
}

void test_linked_list_run_all(void) {
	RUN_TEST(insert_nodes);
	RUN_TEST(remove_nodes);
}
