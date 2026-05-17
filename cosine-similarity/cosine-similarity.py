import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    """
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)

    if a.ndim != 1 or b.ndim != 1 or len(a) != len(b):
        return None

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    similarity = np.dot(a, b) / (norm_a * norm_b)
    return float(similarity)
