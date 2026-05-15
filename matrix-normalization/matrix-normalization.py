import numpy as np

def matrix_normalization(matrix, axis=None, norm_type='l2'):
    """
    Normalize a 2D matrix along specified axis using specified norm.
    """
    eps=1e-12
    matrix = np.array(matrix)
    if matrix.ndim != 2:
        return None
    if axis is not None and (axis<0 or axis>= matrix.ndim):
        return None
    try:
        # Write code here
        if norm_type == 'l2':
            squared = np.square(matrix)
            add = np.sum(squared, axis=axis, keepdims=True)
        
            l2 = np.sqrt(np.maximum(add, eps))
        
            return matrix/l2
    
        elif norm_type == 'l1':
            add = np.sum(np.abs(matrix), axis=axis, keepdims=True)
            
            
            return matrix/(add+eps)
    
        elif norm_type == 'max':
            maximum = np.max(np.abs(matrix), axis=axis, keepdims=True)
            
            
            return matrix/(maximum+eps)

    except Exception:
        return None