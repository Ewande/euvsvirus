import numpy as np


def estimate_pupils_skill(array):
    # limits of level difficulty for each test
    multiplier = np.linspace(0, 1, num=array.shape[1]+1)[1:]
    #returning max in each section scaled by difficulty levels
    return np.max(np.multiply(array, multiplier), axis=1)

# arr = np.array([[0, 0, 0, 0.7], [0.6, 0.8, 0, 0], [0.3, 0, 0.1, 0]]).T
# test = estimate_pupils_skill(arr)
# print(42)