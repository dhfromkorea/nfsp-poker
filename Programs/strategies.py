"""
Contains strategy functions

They should all have the same signature, but don't always use all this information
"""
from game_utils import *
from state_abstraction import build_state
from utils import softmax
import numpy as np


def strategy_random(player, board, pot, actions, b_round, opponent_stack, opponent_side_pot, greedy=True, blinds=BLINDS, verbose=False):
    """
    Take decision using Q values (in a greedy or random way)
    :param player:
    :param board:
    :param pot:
    :param actions:
    :param b_round:
    :param opponent_stack:
    :param Q: the Keras neural network that takes states as inputs
    :param greedy: True for greedy, False for Q-softmax sampling
    :param blinds:
    :param verbose:
    :return:
    """
    possible_actions = authorized_actions_buckets(player, actions, b_round, opponent_side_pot)  # you don't have right to take certain actions, e.g betting more than you have or betting 0 or checking a raise
    random_action_bucket = np.random.choice(possible_actions)
    random_action = bucket_to_action(random_action_bucket, actions, b_round, player, opponent_side_pot)
    return random_action


def strategy_RL_aux(player, board, pot, actions, b_round, opponent_stack, opponent_side_pot, Q, greedy=True, blinds=BLINDS, verbose=False):
    # @todo: add a param eps for eps-greedy policies ?
    """
    Take decision using Q values (in a greedy or random way)
    :param player:
    :param board:
    :param pot:
    :param actions:
    :param b_round:
    :param opponent_stack:
    :param Q: the Keras neural network that takes states as inputs
    :param greedy: True for greedy, False for Q-softmax sampling
    :param blinds:
    :param verbose:
    :return:
    """
    possible_actions = authorized_actions_buckets(player, actions, b_round, opponent_side_pot)  # you don't have right to take certain actions, e.g betting more than you have or betting 0 or checking a raise
    print(player.stack, player.side_pot, opponent_side_pot, possible_actions)
    state = build_state(player, board, pot, actions, b_round, opponent_stack, blinds)
    state = [s.reshape(tuple([1] + list(s.shape))) for s in state]
    Q_values = Q.predict(state)[0].squeeze()  # it has multiple outputs, the first is the Qvalues
    print([float('%.1f' % q) for q in Q_values])

    # choose action in a greedy way
    if greedy:
        Q_values_for_possible_actions = [Q_value for k, Q_value in enumerate(Q_values) if idx_to_bucket(k) in possible_actions]
        best_possible_action_bucket = np.argmax(Q_values_for_possible_actions)
        best_possible_action_bucket = [idx_to_bucket(k) for k, Q_value in enumerate(Q_values) if idx_to_bucket(k) in possible_actions][best_possible_action_bucket]
        action = bucket_to_action(best_possible_action_bucket, actions, b_round, player, opponent_side_pot)
    else:
        idx = [idx_to_bucket(k) for k, Q_value in enumerate(Q_values) if idx_to_bucket(k) in possible_actions]
        Q_values = [Q_value for k, Q_value in enumerate(Q_values) if idx_to_bucket(k) in possible_actions]
        probabilities = softmax(Q_values)
        assert np.abs(np.sum(probabilities) - 1.) < 1e-6, probabilities
        action = bucket_to_action(sample_action(idx, probabilities), actions, b_round, player, opponent_side_pot)
    return action


def strategy_RL(Q, greedy):
    """Function generator"""
    return lambda player, board, pot, actions, b_round, opponent_stack, opponent_side_pot, blinds=BLINDS, verbose=False: strategy_RL_aux(player, board, pot, actions, b_round, opponent_stack, opponent_side_pot, Q, greedy=greedy, blinds=blinds, verbose=verbose)
