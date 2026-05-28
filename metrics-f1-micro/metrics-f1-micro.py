import numpy as np

def f1_micro(y_true, y_pred) -> float:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    tp = np.sum(y_true == y_pred)
    fp = np.sum(y_pred != y_true)
    fn = fp 
    return 2 * tp / (2 * tp + fp + fn)
