# Original: https://github.com/ooki/green_tsetlin/blob/master/generator_tests/create_mnist_test_data.py

import os
import sys
import green_tsetlin as gt

from get_data import get_mnist_data
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src/python')))
from gt_to_bin import save_to_bin


if __name__ == "__main__":
    x, y = get_mnist_data()

    n_clauses = 1000
    n_literals = 784
    n_classes = 10
    s = 10.0
    n_literal_budget = 8
    threshold = 1000
    n_jobs = 2
    seed = 42

    tm = gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s,
                           threshold=threshold, literal_budget=n_literal_budget)

    tm.load_state("data/demos/mnist/mnist_state.npz")

    save_to_bin(tm, "data/models/mnist_tm.bin")

    correct = 0
    total = 0
    p = tm.get_predictor()
    for k in range(0, x.shape[0]):
        y_hat = p.predict(x[k, :])
        if y_hat == y[k]:
            correct += 1

        total += 1

    print("correct:", correct, "total:", total)
    print("Note that the pretrained model is evaluated on training data (as per green_tsetlin example).")
