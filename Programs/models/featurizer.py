# we are going to train featurizer here
# ideally we move all featurizer-related stuff here

import numpy as np
import torch as t
from IPython import display
import pickle
from tqdm import tqdm
import os


from models.q_network import CardFeaturizer11
from game.utils import variable, moving_avg
from game.game_utils import Card, cards_to_array
from matplotlib import pyplot as plt
from sklearn.utils import shuffle

HS_DATA_PATH = 'new_hand_eval/'

train_filenames = [
    'hand_eval_6100000.pkl',
    'hand_eval_2.p',
    'hand_eval_6200000.pkl',
    'hand_eval_6300000.pkl',
    'hand_eval_51511238591.3367474.p',
    'hand_eval_51511242191.357451.p',
    'hand_eval_51511245791.3903465.p',
    'hand_eval_51511251739.1193094.p',
    'hand_eval_51511255339.1697764.p',
    'hand_eval_51511258939.2093961.p',
    'hand_eval_51511262539.280468.p',
    'hand_eval_51511266139.357883.p',
    'hand_eval_51511269739.3839424.p',
    'hand_eval_51511273339.396154.p',
    'hand_eval_51511276939.4614336.p',
    'hand_eval_51511280539.5224082.p',
    'hand_eval_sample71511317987.177001.p'
]
test_filenames = [
    'hand_eval_6400000.pkl', 
    'hand_eval_51511291339.6279852.p'
]
