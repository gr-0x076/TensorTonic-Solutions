import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    y = np.array(y)
    
    _, counts = np.unique(y, return_counts=True)
    
    p = counts / counts.sum()
    
    entropy = -np.sum(p * np.log2(p, where=(p > 0)))
    
    return entropy
