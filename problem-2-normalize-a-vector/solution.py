import numpy as np


def solve(x):
    """Implement Normalize a Vector from the mathematical contract in README.md."""
    x = np.array(x)
    x_norm_2 = np.linalg.norm(x, ord=2)
    if np.array_equal(x, np.zeros_like(x)):
        raise ValueError("not valid input")
    return x / x_norm_2
