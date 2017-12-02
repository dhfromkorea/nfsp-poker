import numpy as np
import torch as t
from torch.autograd import Variable
from game.game_utils import cards_to_array, actions_to_array
from game.utils import variable

def create_state_variable(state, cuda=False):
    # TODO: check if dtype should be handled individually
    # do not use this. use variable() in utils
    f = lambda x: variable(x, cuda=cuda)
    return [f(e) for e in state]


def create_state_variable_batch():
    f = np.vectorize(create_state_variable)
    return f


def create_state_vars_batch(states_batch):
    '''
    states_batch: np array of batch_size x number of features in state
    '''
    dtype = t.FloatTensor
    num_features = states_batch.shape[1]
    state_vars = []
    for i in range(num_features):
        var = Variable(t.from_numpy(states_batch[:, i]).type(dtype), requires_grad=False)
        state_vars.append(var)
    return state_vars


def build_state(player, board, pot, actions, opponent_stack, big_blind, as_variable=False):
    # @todo: add opponent modeling
    """
    Return state as numpy arrays (inputs of Q networks)
        - hand
        - board
        - pot - stack - opponent stack - blinds
        - dealer
        - opponent model
        - preflop plays
        - flop plays
        - turn plays
        - river plays
    :param player:
    :param board:
    :param pot:
    :param actions:
    :param b_round:
    :param opponent_stack:
    :param blinds:
    :param as_variable: torch
    :return:
    """
    hand = cards_to_array(player.cards)
    board = cards_to_array(board)
    pot_ = np.array([pot])
    stack_ = np.array([player.stack])
    opponent_stack_ = np.array([opponent_stack])
    big_blind_ = np.array([big_blind])
    dealer = np.array([player.id if player.is_dealer else 1 - player.id])
    preflop_plays, flop_plays, turn_plays, river_plays = actions_to_array(actions)

    state = [hand, board, pot_, stack_, opponent_stack_, big_blind_, dealer, preflop_plays, flop_plays, turn_plays, river_plays]
    state = [np.expand_dims(s, axis=0) for s in state]
    if as_variable:
        return create_state_variable(state)
    else:
        return state
