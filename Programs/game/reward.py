import numpy as np
import torch as t
from torch.autograd import Variable


def create_reward_variable(reward):
    # TODO: check if dtype should be handled individually
    dtype = t.FloatTensor
    return Variable(t.from_numpy(np.array([reward])).type(dtype), requires_grad=False)


def create_reward_variable_batch():
    f = np.vectorize(create_reward_variable)
    return f
