# Tsetlin Machine C
C library for Tsetlin Machines.

## Features
- C library for Tsetlin Machines: inference, training, saving to / loading from bin files
- TM types: normal (dense), sparse, stateless (sparse)
- model import from green_tsetlin https://github.com/ooki/green_tsetlin

## Requirements
- gcc
- make (optional)
- python (optional - importing models from green_tsetlin, demo data)
- uv (optional - if importing models from green_tsetlin, demo data)

## Install
- `git clone https://github.com/notEloiir/green_tsetlin_to_trainable_C.git`
- `cd green_tsetlin_to_trainable_C`
- `uv sync` (if using uv)

## Run demos
- MNIST training, data downloaded by python script
    - `uv run make run_mnist_demo`
    - `make run_mnist_demo_c` after first run
- MNIST inference using pretrained (dense) model, test data downloaded by python script
    - `uv run make run_mnist_inference_demo`
    - `make run_mnist_inference_demo_c` after first run
- Model size comparison of different TM types loading a pretrained (dense) model
    - `make run_model_size_demo`

## Run tests
- `make run_tests`
