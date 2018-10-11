import numpy as np


def get_liner_se(theta, scale):

    rect = np.zeros((scale, scale), dtype=np.float32)
    if theta == 0:
        rect = np.ones((1,scale), dtype=np.float32)
    elif theta == 90:
        rect = np.ones((scale, 1), dtype=np.float32)
    elif theta == 45:
        for i in range(scale):
            rect[i, scale-1-i] = 1
    elif theta == 135:
        for i in range(scale):
            rect[i, i] = 1
    else:
        raise ValueError(' theta must in (0, 90, 45, 135)')

    return rect
