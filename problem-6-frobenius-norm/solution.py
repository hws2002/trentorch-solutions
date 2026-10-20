import numpy as np


def solve(A):
    # TODO: Return the Frobenius norm of A.
    A = np.array(A).flatten()
    return np.linalg.norm(A,ord=2)
