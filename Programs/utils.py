import numpy as np


def softmax(x):
    p = np.exp(x) / np.sum(np.exp(x))
    # p = p*(p >= 1e-3)
    # for k, x in enumerate(p):
    #     p[k] = int(10000*x)/10000
    # p /= p.sum()
    return p


def sample_categorical(probabilities):
    stops = [0]
    for p in probabilities:
        stops.append(stops[-1]+p)
    u = np.random.uniform()
    for k in range(len(stops)-1):
        if stops[k] <= u < stops[k+1]:
            return k
    raise ValueError('It should have returned something')

