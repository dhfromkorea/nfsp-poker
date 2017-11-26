import numpy as np
import torch as t
import datetime
import os, errno


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


def variable(array, requires_grad=False, to_float=True, cuda=False):
    """Wrapper for t.autograd.Variable"""
    if isinstance(array, np.ndarray):
        v = t.autograd.Variable(t.from_numpy(array), requires_grad=requires_grad)
    elif isinstance(array, list) or isinstance(array,tuple):
        v = t.autograd.Variable(t.from_numpy(np.array(array)), requires_grad=requires_grad)
    elif isinstance(array, float) or isinstance(array, int):
        v = t.autograd.Variable(t.from_numpy(np.array([array])), requires_grad=requires_grad)
    elif isinstance(array, t.Tensor):
        v = t.autograd.Variable(array, requires_grad=requires_grad)
    else: raise ValueError
    if cuda:
        v = v.cuda()
    if to_float:
        return v.float()
    else:
        return v


def moving_avg(x, window=50):
    return [np.mean(x[k:k+window]) for k in range(len(x)-window)]


def initialize_save_folder(path):
    date = datetime.datetime.now().strftime('%Y_%m_%d')
    save_path = path + date + '/'
    try:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            if not os.path.exists(save_path + 'img/'):
                os.makedirs(save_path + 'img/')
            if not os.path.exists(save_path + 'saved_models/'):
                os.makedirs(save_path + 'saved_models/')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise Exception('could not initialize save data folder')
    return save_path
