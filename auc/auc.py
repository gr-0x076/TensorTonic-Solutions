import numpy as np

def auc(fpr, tpr):
    """
    Compute AUC (Area Under ROC Curve) using trapezoidal rule.
    """
    fpr = np.array(fpr)
    tpr = np.array(tpr)
    auc_value = 0.0
    for i in range(1, len(fpr)):
        auc_value += (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2.0
    return auc_value
