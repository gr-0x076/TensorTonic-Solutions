import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    positions = np.arange(seq_len)[:, np.newaxis]        
    dims = np.arange(d_model)[np.newaxis, :]             
    angle_rates = 1 / np.power(base, (2 * (dims // 2)) / d_model)
    angle_rads = positions * angle_rates

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return angle_rads
