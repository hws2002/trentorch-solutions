import numpy as np


def solve(X):
    # TODO: Return the Gram matrix X^T X.
    X = np.array(X)
    return X.T @ X
