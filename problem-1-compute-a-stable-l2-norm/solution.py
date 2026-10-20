import numpy as np


def solve(x):
    """Implement Compute a Stable L2 Norm from the mathematical contract in README.md."""
    # return np.linalg.norm(x, ord=2) is unstable
    x = np.array(x)
    max_val = np.max(abs(x))
    return max_val * np.linalg.norm(x / max_val,ord=2)
    return max_val
