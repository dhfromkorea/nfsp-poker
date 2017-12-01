import numpy as np
import torch as t
import datetime
import os, errno


def get_last_round(actions, player):
    for i in reversed(range(0, 4)):
        if len(actions[i][player])>0:
            return i
    return -1


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


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
