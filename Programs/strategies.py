"""
Contains strategy functions

They should all have the same signature, but don't always use all this information
"""
from game_utils import *
from state_abstraction import build_state
from utils import softmax
import numpy as np


def strategy_limper(player, board, pot, actions, b_round, opponent_stack, opponent_side_pot, blinds=BLINDS, verbose=False):
    """
    The limper never raises but check and call often
    :param player: the player
    :param board: the cards of the board (a list of them)
    :param pot: the value of the pot
    :param actions: the actions that were taken up to now {betting_round: {player: [actions_it_took]}}
    :param b_round: betting round id (0 for preflop, 1 for flop...)
    :param opponent_stack: the stack of the opponent
    :param verbose: whether to print stuff or not
    :return: the action of the limper
    """
    # get some info about the player
    player_name = str(player.id) if player.name is None else player.name
    id = player.id
    is_dealer = player.is_dealer
    cards = player.cards

    # the id of the opponent
    opponent = 1 - id

    # treat differently preflop from other rounds
    if b_round > 0:
        # if the opponent didn't play yet, let's bet
        if len(actions[b_round][opponent]) == 0:
            if player.stack > BLINDS[1]:
                if verbose:
                    if player_name is None:
                        player_name = str(id)
                    print(player_name + ' (' + str(player.stack) + ') ' + 'bet ' + str(BLINDS[1]))
                return Action('bet', BLINDS[1])
            else:
                if verbose:
                    if player_name is None:
                        player_name = str(id)
                    print(player_name + ' (' + str(player.stack) + ') '+ ' is all in (' + str(player.stack)+')')
                return Action('all in', player.stack)
        else:  # otherwise, check or call
            if actions[b_round][opponent][-1].type == 'check':
                if verbose:
                    if player_name is None:
                        player_name = str(id)
                    print(player_name + ' checked')
                return Action('check')
            else:
                if actions[b_round][opponent][-1].value < 10*BLINDS[1]:
                    if player.stack > actions[b_round][opponent][-1].value:
                        if verbose:
                            if player_name is None:
                                player_name = str(id)
                            print(player_name + ' called')
                        return Action('call', actions[b_round][opponent][-1].value)
                    else:
                        if verbose:
                            if player_name is None:
                                player_name = str(id)
                            print(player_name + ' is all in ('+str(player.stack)+')')
                        return Action('all in', player.stack)
                else:  # if the opponent bet too much, fold
                    if verbose:
                        if player_name is None:
                            player_name = str(id)
                        print(player_name + ' folded')
                    return Action('fold')
    else:
        is_SB = is_dealer
        if is_SB:
            # if you have enough money to call the BB, call it
            if player.stack > BLINDS[0]:
                if verbose:
                    if player_name is None:
                        player_name = str(id)
                    print(player_name + ' (' + str(player.stack) + ') ' + ' called (' + str(BLINDS[0]) + ')')
                return Action('call', BLINDS[0])

            # otherwise, go all in
            elif 0 < player.stack <= BLINDS[0]:
                if verbose:
                    if player_name is None:
                        player_name = str(id)
                    print(player_name + ' (' + str(player.stack) + ') ' + ' is all in (' + str(player.stack)+')')
                return Action('all in', player.stack)

            # you may already be all-in. In this case, that should be treated before, showdown
            else:
                return Action('null')
        else:
            # if the opponent called your BB, check
            if actions[0][opponent][-1].type == 'call':
                return Action('check')

            # if he raised not too much, call if you have enough money, otherwise go all in
            if actions[0][opponent][-1].type == 'raise':
                if actions[0][opponent][-1].value > 10*BLINDS[1]:
                    if player.stack > actions[0][opponent][-1].value:
                        return Action('call', actions[0][opponent][-1].value - BLINDS[1] + BLINDS[0])
                    elif 0 < player.stack <= actions[0][opponent][-1].value:
                        return Action('all in', player.stack)
                    # this case should be treated before
                    else:
                        return Action('null')
                else:
                    return Action('fold')
            if actions[0][opponent][-1].type == 'all in':
                if actions[0][opponent][-1].value > 10*BLINDS[1]:
                    if player.stack > actions[0][opponent][-1].value:
                        return Action('call', actions[0][opponent][-1].value - BLINDS[1] + BLINDS[0])
                    elif 0 < player.stack <= actions[0][opponent][-1].value:
                        return Action('all in', player.stack)
                    # this case should be treated before
                    else:
                        return Action('null')
                else:
                    return Action('fold')


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
    state = build_state(player, board, pot, actions, b_round, opponent_stack, blinds)
    state = [s.reshape(tuple([1] + list(s.shape))) for s in state]
    Q_values = Q.predict(state)[0].squeeze()  # it has multiple outputs, the first is the Qvalues

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
    if verbose:
        print(player.name + ' (' + str(player.stack) + ') ' + action.type + '(' + str(action.value) + ')')
    return action


def strategy_RL(Q, greedy):
    """Function generator"""
    return lambda player, board, pot, actions, b_round, opponent_stack, opponent_side_pot, blinds=BLINDS, verbose=False: strategy_RL_aux(player, board, pot, actions, b_round, opponent_stack, opponent_side_pot, Q, greedy=greedy, blinds=blinds, verbose=verbose)
