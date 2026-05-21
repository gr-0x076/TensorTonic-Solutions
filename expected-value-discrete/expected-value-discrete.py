import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    Raises ValueError if probabilities are invalid
    """
    x = np.array(x, dtype=float)
    p = np.array(p, dtype=float)

    if not np.isclose(np.sum(p), 1.0):
        raise ValueError("Probabilities must sum to 1")

    return np.sum(x * p)
