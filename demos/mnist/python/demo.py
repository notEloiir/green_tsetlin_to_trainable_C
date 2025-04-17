# Original: https://github.com/ooki/green_tsetlin/blob/master/generator_tests/create_mnist_test_data.py

import os
import sys
import green_tsetlin as gt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src/python')))
from gt_to_bin import save_to_bin


if __name__ == "__main__":
    fetch_mnist = False
    if fetch_mnist:
        X, y = fetch_openml(
            "mnist_784",
            version=1,
            return_X_y=True,
            as_frame=False)

        x, y = shuffle(X, y, random_state=42)
        x = np.where(x.reshape((x.shape[0], 28 * 28)) > 75, 1, 0)
        x = x.astype(np.uint8)
        y = y.astype(np.uint32)

        n_examples = x.shape[0]  # 70000
        n_literals = x.shape[1]  # 784
        x.astype(np.uint8).tofile("data/demos/mnist/mnist_x_{}_{}.test_bin".format(n_examples, n_literals))
        y.astype(np.uint32).tofile("data/demos/mnist/mnist_y_{}_{}.test_bin".format(n_examples, n_literals))
    else:
        x = np.fromfile("data/demos/mnist/mnist_x_70000_784.test_bin", dtype=np.uint8)
        y = np.fromfile("data/demos/mnist/mnist_y_70000_784.test_bin", dtype=np.uint32)
        x = x.reshape((70000, 784))

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
    correct2 = 0
    total = 0
    p = tm.get_predictor()
    for k in range(0, x.shape[0]):
        y_hat = p.predict(x[k, :])
        if y_hat == y[k]:
            correct += 1

        total += 1

    print("correct:", correct, "correct2:", correct2, "total:", total)
    print("<done>")
