import numpy as np

def solve(x):
    """Implement min-max-feature-scaling from the mathematical contract in README.md."""
    if np.max(x) == np.min(x):
        return x - np.max(x)
    return (x - np.min(x))/(np.max(x) - np.min(x))
