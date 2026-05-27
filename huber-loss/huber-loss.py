import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    e = y_true - y_pred
    abs_e = np.abs(e)

    quadratic = np.minimum(abs_e, delta)
    linear = abs_e - quadratic

    loss = 0.5 * quadratic**2 + delta * linear
    return np.mean(loss)
