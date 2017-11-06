import numpy as np
from state_abstraction import cards_to_array, actions_to_array, action_to_array
from game_utils import BLINDS


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
