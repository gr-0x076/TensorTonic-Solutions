import numpy as np

def ridge_regression(X, y, lam=0.0):
    """
    Compute ridge regression weights using the closed-form solution.
    Falls back to pseudo-inverse if matrix is singular.
    """
    X = np.array(X)
    y = np.array(y)
    I = np.eye(X.shape[1])

    if lam == 0:
        try:
            theta = np.linalg.inv(X.T @ X) @ X.T @ y
        except np.linalg.LinAlgError:
            theta = np.linalg.pinv(X.T @ X) @ X.T @ y
    else:
        try:
            theta = np.linalg.inv(X.T @ X + lam * I) @ X.T @ y
        except np.linalg.LinAlgError:
            theta = np.linalg.pinv(X.T @ X + lam * I) @ X.T @ y

    return theta
