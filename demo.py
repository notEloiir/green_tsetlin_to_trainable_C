import green_tsetlin as gt
import numpy as np


def save_to_bin(tm: gt.TsetlinMachine, filename: str):
    threshold: int = tm.threshold
    n_features: int = tm.n_literals
    n_clauses: int = tm.n_clauses
    n_classes: int = tm.n_classes
    max_state: int = 127    # hardcoded in green_tsetlin/src/func_tm.hpp
    min_state: int = -127   # hardcoded in green_tsetlin/src/func_tm.hpp
    boost_true_positive_feedback: int = int(tm.boost_true_positives)

    weights: np.ndarray = tm._state.w  # shape=(n_clauses, n_classes), dtype=np.int16
    clauses: np.ndarray = tm._state.c  # shape=(n_clauses, n_literals*2), dtype=np.int8

    print("threshold", threshold,
          "n_features", n_features,
          "n_clauses", n_clauses,
          "n_classes", n_classes,
          "max_state", max_state,
          "min_state", min_state,
          "boost_true_positive_feedback", boost_true_positive_feedback,
          "weights", weights.shape, weights,
          "clauses", clauses.shape, clauses,
          sep='\n')

    with open(filename, "wb") as f:
        # Write metadata
        f.write(threshold.to_bytes(4, "little", signed=True))
        f.write(n_features.to_bytes(4, "little", signed=True))
        f.write(n_clauses.to_bytes(4, "little", signed=True))
        f.write(n_classes.to_bytes(4, "little", signed=True))
        f.write(max_state.to_bytes(4, "little", signed=True))
        f.write(min_state.to_bytes(4, "little", signed=True))
        f.write(boost_true_positive_feedback.to_bytes(4, "little", signed=True))

        # Write weights and clauses
        f.write(weights.astype(np.int16).tobytes())
        f.write(clauses.astype(np.int8).tobytes())


if __name__ == "__main__":
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
