import numpy as np

def nesterov_momentum_step(w, v, grad, lr=0.1, momentum=0.9):
    """
    Perform one Nesterov Momentum update step.
    """
    w = np.array(w, dtype=float)
    v = np.array(v, dtype=float)
    grad = np.array(grad, dtype=float)

    w_look = w - momentum * v


    v_new = momentum * v + lr * grad

    w_new = w - v_new

    return w_new, v_new

