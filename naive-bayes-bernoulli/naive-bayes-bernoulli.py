import numpy as np

def naive_bayes_bernoulli(X_train, y_train, X_test):
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)

    classes = np.unique(y_train)
    n_features = X_train.shape[1]


    priors = {c: np.mean(y_train == c) for c in classes}

    probs = {}
    for c in classes:
        X_c = X_train[y_train == c]
        probs[c] = (np.sum(X_c, axis=0) + 1) / (X_c.shape[0] + 2)

    results = []
    for x in X_test:
        ll = []
        for c in classes:
            p = probs[c]
            prior = np.log(priors[c])
            feature_ll = np.sum(x * np.log(p) + (1 - x) * np.log(1 - p))
            ll.append(prior + feature_ll)
        results.append(ll)
    return results
