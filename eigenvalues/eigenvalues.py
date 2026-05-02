import numpy as np

def calculate_eigenvalues(matrix):
    try:
        arr = np.array(matrix, dtype=float)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            return None
        vals = np.linalg.eigvals(arr)
        return np.sort(vals)
    except Exception:
        return None
