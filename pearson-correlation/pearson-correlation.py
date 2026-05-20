import numpy as np

def pearson_correlation(X):
    X = np.array(X, dtype=float)
    n = X.shape[0]
    

    X_centered = X - X.mean(axis=0)
    
    cov = (X_centered.T @ X_centered) / (n - 1)
    
    std = X.std(axis=0, ddof=1)
    
    corr = cov / np.outer(std, std)
    
    return corr
