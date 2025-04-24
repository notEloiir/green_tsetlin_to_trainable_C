.PHONY: all run_demo_py run_demo_c clean

CC = gcc
CFLAGS = -Wall -Wextra -O2
C_SRC = src/c/src/tsetlin_machine.c src/c/src/tm_output_activation.c src/c/src/utility.c
C_TESTS_SRC = tests/c/test_tsetlin_machine.c tests/c/unity/unity.c
BUILD_DIR = build
INCLUDE = -I src/c/include


# === Default target ===
all: run_demo

# === File targets ===
mnist_demo: $(C_SRC) demos/mnist/c/demo.c
	mkdir -p $(BUILD_DIR)
	$(CC) $(INCLUDE) $(CFLAGS) $^ -o $(BUILD_DIR)/$@

tests: $(C_SRC) $(C_TESTS_SRC)
	mkdir -p $(BUILD_DIR)
	$(CC) $(INCLUDE) $(CFLAGS) $^ -o $(BUILD_DIR)/$@

# === Run targets ===
run_demo: run_demo_py run_demo_c

run_demo_c: mnist_demo
	./$(BUILD_DIR)/mnist_demo

run_demo_py:
	poetry run python demos/mnist/python/demo.py

run_tests: run_tests_c

run_tests_c: tests
	./$(BUILD_DIR)/tests

# === Cleanup ===
clean:
	rm -rf $(BUILD_DIR)/* src/python/__pycache__
