.PHONY: all run_demo_py run_demo_c clean

C_SRC = src/c/src/tsetlin_machine.c
BUILD_DIR = build
INCLUDE = -I src/c/include


# === Default target ===
all: run_demo_py run_demo_c

# === File targets ===
mnist_demo: $(C_SRC) demos/mnist/c/demo.c
	mkdir -p $(BUILD_DIR)
	gcc $(INCLUDE) -Wall -Wextra -O2 $^ -o $(BUILD_DIR)/$@

# === Run targets ===
run_demo_c: mnist_demo
	./$(BUILD_DIR)/mnist_demo

run_demo_py:
	poetry run python demos/mnist/python/demo.py

# === Cleanup ===
clean:
	rm -rf $(BUILD_DIR)/* rmdir src/python/__pycache__
