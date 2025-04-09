all: run_demo_py run_demo_c clean

demo: demo.c
	gcc -Wall -Wextra -O2 demo.c TsetlinMachine.c -o demo

run_demo_c: demo
	./demo

run_demo_c_clean: run_demo_c clean

run_demo_py: demo.py
	poetry run python demo.py

clean:
	rm ./demo

