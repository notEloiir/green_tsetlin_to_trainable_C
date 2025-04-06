# Original: https://github.com/ooki/green_tsetlin/blob/master/generator_tests/create_mnist_test_data.py

import green_tsetlin as gt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle


def save_to_bin(tm: gt.TsetlinMachine, filename: str):
    threshold: int = tm.threshold
    n_literals: int = tm.n_literals
    n_clauses: int = tm.n_clauses
    n_classes: int = tm.n_classes
    max_state: int = 127    # hardcoded in green_tsetlin/src/func_tm.hpp
    min_state: int = -127   # hardcoded in green_tsetlin/src/func_tm.hpp
    boost_true_positive_feedback: int = int(tm.boost_true_positives)

    weights: np.ndarray = tm._state.w  # shape=(n_clauses, n_classes), dtype=np.int16
    clauses: np.ndarray = tm._state.c  # shape=(n_clauses, n_literals*2), dtype=np.int8
    weights_transposed = weights.transpose()  # shape=(n_classes, n_clauses), dtype=np.int16
    clauses_reordered = clauses.reshape(n_clauses, 2, n_literals).transpose(0, 2, 1).reshape(n_clauses, -1)

    print("threshold", threshold,
          "n_literals", n_literals,
          "n_clauses", n_clauses,
          "n_classes", n_classes,
          "max_state", max_state,
          "min_state", min_state,
          "boost_true_positive_feedback", boost_true_positive_feedback,
          "weights (transposed)", weights_transposed.shape, weights_transposed,
          "clauses (reordered)", clauses_reordered.shape, clauses_reordered,
          sep='\n')

    with open(filename, "wb") as f:
        # Write metadata
        f.write(threshold.to_bytes(4, "little", signed=True))
        f.write(n_literals.to_bytes(4, "little", signed=True))
        f.write(n_clauses.to_bytes(4, "little", signed=True))
        f.write(n_classes.to_bytes(4, "little", signed=True))
        f.write(max_state.to_bytes(4, "little", signed=True))
        f.write(min_state.to_bytes(4, "little", signed=True))
        f.write(boost_true_positive_feedback.to_bytes(4, "little", signed=True))

        # Write weights and clauses
        f.write(weights_transposed.astype(np.int16).tobytes())
        f.write(clauses_reordered.astype(np.int8).tobytes())


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
        x.astype(np.uint8).tofile("./mnist_x_{}_{}.test_bin".format(n_examples, n_literals))
        y.astype(np.uint32).tofile("./mnist_y_{}_{}.test_bin".format(n_examples, n_literals))
    else:
        x = np.fromfile("./mnist_x_70000_784.test_bin", dtype=np.uint8)
        y = np.fromfile("./mnist_y_70000_784.test_bin", dtype=np.uint32)
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

    tm.load_state("mnist_state.npz")

    save_to_bin(tm, "mnist_tm.bin")

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
