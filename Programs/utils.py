import numpy as np


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))