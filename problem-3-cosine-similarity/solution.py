import numpy as np


def solve(a, b):
    """Implement Cosine Similarity from the mathematical contract in README.md."""
    a = np.array(a)
    b = np.array(b)
    a /= np.linalg.norm(a, ord = 2)
    b /= np.linalg.norm(b, ord = 2)
    return np.sum(a * b)
