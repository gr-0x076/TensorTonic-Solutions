import numpy as np

def apply_homogeneous_transform(T, points):
    T = np.array(T)
    points = np.array(points)

    points = np.atleast_2d(points)

    ones = np.ones((points.shape[0], 1))
    homo_points = np.hstack([points, ones])

    transformed = homo_points @ T.T

    result = transformed[:, :3]

    return result[0] if points.shape[0] == 1 else result
