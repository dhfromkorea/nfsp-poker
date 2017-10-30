from utils import *


def strategy_limper(id, is_dealer, cards, board, pot, actions, b_round, opponent_stack, verbose=False, player_name=None):
    opponent = (id+1)%2
    if len(actions[b_round][opponent]) == 0:
        if verbose:
            if player_name is None:
                player_name = str(id)
            print(player_name + ' bet ' + str(BLINDS[1]))
        return Action('bet', BLINDS[1])
    else:
        if actions[b_round][opponent][-1].type == 'check':
            if verbose:
                if player_name is None:
                    player_name = str(id)
                print(player_name + ' checked')
            return Action('check')
        else:
            if actions[b_round][opponent][-1].value < 10*BLINDS[1]:
                if verbose:
                    if player_name is None:
                        player_name = str(id)
                    print(player_name + ' called')
                return Action('call', actions[b_round][opponent][-1].value)
            else:
                if verbose:
                    if player_name is None:
                        player_name = str(id)
                    print(player_name + ' folded')
                return Action('fold')
