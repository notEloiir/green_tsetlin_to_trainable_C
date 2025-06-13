.PHONY: all run_demo_py run_demo_c clean

CC = gcc
CFLAGS = -Wall -Wextra -O2
C_SRC = src/c/src/fast_prng.c src/c/src/utility.c src/c/src/tsetlin_machine.c src/c/src/sparse_tsetlin_machine.c src/c/src/stateless_tsetlin_machine.c 
C_TESTS_SRC = tests/c/unity/unity.c tests/c/test_runner.c tests/c/test_tsetlin_machine.c tests/c/test_linked_list.c
BUILD_DIR = build
INCLUDE = -I src/c/include


# === Default target ===
all: run_mnist_demo

# === File targets ===
mnist_demo_inference: $(C_SRC) demos/mnist/c/mnist_util.c demos/mnist/c/pretrained_inference_demo.c
	mkdir -p $(BUILD_DIR)
	$(CC) $(INCLUDE) $(CFLAGS) $^ -o $(BUILD_DIR)/$@

mnist_demo: $(C_SRC) demos/mnist/c/mnist_util.c demos/mnist/c/training_demo.c
	mkdir -p $(BUILD_DIR)
	$(CC) $(INCLUDE) $(CFLAGS) $^ -o $(BUILD_DIR)/$@

model_size_demo: $(C_SRC) demos/model_size/c/demo.c
	mkdir -p $(BUILD_DIR)
	$(CC) $(INCLUDE) $(CFLAGS) $^ -o $(BUILD_DIR)/$@

tests_bin: $(C_TESTS_SRC)
	mkdir -p $(BUILD_DIR)
	$(CC) $(INCLUDE) $(CFLAGS) $^ -o $(BUILD_DIR)/$@

# === Run targets ===
run_mnist_inference_demo: run_mnist_inference_demo_py run_mnist_inference_demo_c

run_mnist_inference_demo_c: mnist_demo_inference
	./$(BUILD_DIR)/mnist_demo_inference

run_mnist_inference_demo_py:
	python demos/mnist/python/get_pretrained.py

run_mnist_demo: run_mnist_demo_py run_mnist_demo_c

run_mnist_demo_c: mnist_demo
	./$(BUILD_DIR)/mnist_demo

run_mnist_demo_py:
	python demos/mnist/python/get_data.py

run_model_size_demo: run_model_size_demo_c

run_model_size_demo_c: model_size_demo
	./$(BUILD_DIR)/model_size_demo

run_tests: run_tests_c

run_tests_c: tests_bin
	./$(BUILD_DIR)/tests_bin

# === Cleanup ===
clean:
	rm -rf $(BUILD_DIR)/* src/python/__pycache__
