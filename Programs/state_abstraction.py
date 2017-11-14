import numpy as np
from game_utils import cards_to_array, actions_to_array


def build_state(player, board, pot, actions, b_round, opponent_stack, blinds):
    # @todo: add opponent modeling
    """
    Return state as numpy arrays (inputs of Q networks)
        - hand
        - board
        - pot
        - stack
        - opponent stack
        - blinds
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
    :return:
    """
    hand = cards_to_array(player.cards)
    board = cards_to_array(board)
    pot_ = np.array([pot])
    stack_ = np.array([player.stack])
    opponent_stack_ = np.array([opponent_stack])
    blinds_ = np.array(blinds)
    dealer = np.array([player.id if player.is_dealer else 1 - player.id])
    preflop_plays, flop_plays, turn_plays, river_plays = actions_to_array(actions)
    # hand, board, pot, stack, opponent_stack, blinds, dealer, preflop_plays, flop_plays, turn_plays, river_plays]
    return [hand, board, pot_, stack_, opponent_stack_, blinds_, dealer, preflop_plays, flop_plays, turn_plays, river_plays]