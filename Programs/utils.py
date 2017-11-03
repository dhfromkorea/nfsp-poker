import numpy as np
from actions import *
from game_utils import *
from state_abstraction import *


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def update_memory(MEMORY, players, action, new_game, board, pot, dealer, actions):
    player = players[0]
    state_ = [cards_to_array(player.cards), cards_to_array(board), pot, player.stack, players[1].stack,
              np.array(BLINDS), dealer, actions_to_array(actions)]
    action_ = action_to_array(action)
    reward_ = -action.value
    transition = {'s': state_, 'a': action_, 'r': reward_}
    if len(MEMORY) > 0 and not new_game:  # don't take into account transitions overlapping two different games
        # don't forget to store next state
        MEMORY[-1]["s'"] = state_
    MEMORY.append(transition)
