import numpy as np
import torch as t
from torch.autograd import Variable


def create_action_variable(action):
    # TODO: check if dtype should be handled individually
    dtype = t.FloatTensor
    return Variable(t.from_numpy(action).type(dtype), requires_grad=False)


def create_action_variable_batch():
    f = np.vectorize(create_action_variable)
    return f


