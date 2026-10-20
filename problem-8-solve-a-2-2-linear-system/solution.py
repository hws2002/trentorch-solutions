import numpy as np

def solve_linear(A,y):
    """
        return solution of linear system Ax = y.
        Given that is non-singluar.
    """

    # 前代
    n_row, n_col = A.shape[0], A.shape[1]
    for i in range(n_row):
        a = A[i][i]
        for j in range(i+1,n_row):
            factor = A[j][i] / a
            A[j] -= A[i] * factor
            y[j] -= y[i] * factor
    # 后代
    x = np.zeros_like(y)
    for i in range(n_row-1,-1,-1):
        sum = 0
        for j in range(n_row-1, i, -1):
            sum += x[j] * A[i][j]
        x[i] = (y[i] - sum) / A[i][i]
    return x

def solve(coeffs):
    """Solve the system a*x + b*y = e and c*x + d*y = f, where coeffs is [a,b,c,d,e,f]. A singular system raises ValueError."""
    a,b,c,d,e,f = coeffs
    if a*d - b*c == 0:
        raise ValueError('unsolvable!')
    A = np.array([[a, b], [c, d]], dtype=float)
    y = np.array([e, f], dtype=float)
    return solve_linear(A,y)
