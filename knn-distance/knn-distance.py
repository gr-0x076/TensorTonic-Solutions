import numpy as np

def knn_distance(X_train, X_test, k):
    """
    Compute pairwise distances and return k nearest neighbor indices.
    """
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)

    dists = np.sqrt(((X_test[:, None, :] - X_train[None, :, :]) ** 2).sum(axis=2))

    sorted_idx = np.argsort(dists, axis=1)

    n_test = X_test.shape[0]
    neighbors = np.full((n_test, k), -1, dtype=int)

    for i in range(n_test):
        top = sorted_idx[i, :min(k, X_train.shape[0])]
        neighbors[i, :len(top)] = top

    return neighbors
