all: run_demo clean

demo: demo.c
	gcc -Wall -Wextra demo.c -o demo

run_demo: demo.py demo
	poetry run python demo.py
	./demo

clean:
	rm ./demo

