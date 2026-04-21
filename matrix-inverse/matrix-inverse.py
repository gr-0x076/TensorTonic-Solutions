import numpy as np

def matrix_inverse(A):
    """
    Returns: A_inv of shape (n, n) such that A @ A_inv ≈ I
    If matrix is singular or invalid, returns None.
    """
    A = np.array(A)

    try:
        return np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return None
