from time import time
from evaluation import *
from game_utils import Deck, Player, set_dealer, blinds, deal, agreement, split_pot, actions_to_array, action_to_array, cards_to_array
# from q_network import *
from strategies import strategy_RL, strategy_limper, strategy_random
from utils import *
from config import BLINDS


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


verbose = True

# instantiate game
deck = Deck()
INITIAL_MONEY = 100*BLINDS[0]
# players = [Player(0, strategy_limper, INITIAL_MONEY, verbose=True, name='SB'),
#            Player(1, strategy_limper, INITIAL_MONEY, verbose=True, name='DH')]
# Q = Q_network(10, 14)
# players = [Player(0, strategy_RL(Q, False), INITIAL_MONEY, verbose=True, name='SB'),
#            Player(1, strategy_RL(Q, False), INITIAL_MONEY, verbose=True, name='DH')]
players = [Player(0, strategy_random, INITIAL_MONEY, verbose=True, name='SB'),
           Player(1, strategy_random, INITIAL_MONEY, verbose=True, name='DH')]
board = []
dealer = set_dealer(players)
MEMORY = []  # a list of dicts with keys s,a,r,s'
new_game = True
episodes = 0
games = {'n': 0, '#episodes': []}
t0 = time()

while True:
    # at the beginning of a whole new game (one of the player lost or it is the first), all start with the same amounts of money again
    if new_game:
        games['n'] += 1
        games['#episodes'].append(episodes)
        episodes = 0
        if verbose:
            print('####################'
                  'New game (%s) starts.\n'
                  'Players get cash\n'
                  'Last game lasted %.1f\n'
                  'Memory contains %s transitions\n'
                  '####################' % (str(games['n']), time() - t0, str(len(MEMORY))))
            t0 = time()
        players[0].cash(INITIAL_MONEY)
        players[1].cash(INITIAL_MONEY)

    # put blinds
    pot = blinds(players, verbose=verbose)
    if verbose:
        print('pot: ' + str(pot))

    # shuffle decks are clear board
    deck.populate()
    deck.shuffle()
    board = []
    players[0].is_all_in = False
    players[1].is_all_in = False

    # keep track of actions of each player for this episode. Also keep track of the blinds they paid (if you are big blind but have only a small blind, we need to know that you paid only 1 (to compute potential split pot))
    actions = {b_round: {player: [] for player in range(2)} for b_round in range(-1, 4)}
    actions[-1][players[0].id] = players[0].side_pot
    actions[-1][players[1].id] = players[1].side_pot

    # dramatic events monitoring
    fold_occured = False
    null = 0
    all_in = 0  # 0, 1 or 2. If 2, the one of the player is all-in and the other is either all-in or called. In that case, things should be treated differently
    if players[0].stack == 0:  # in this case the blind puts it all-in
        all_in = 2
        players[0].is_all_in = True
    if players[1].stack == 0:
        all_in = 2
        players[1].is_all_in = True

    # betting rounds
    for b_round in range(4):
        # differentiate the case where players are all-in from the one where none of them is
        if all_in < 2:
            # deal cards
            deal(deck, players, board, b_round, verbose=verbose)
            agreed = False  # True when the max bet has been called by everybody

            # play
            if b_round != 0:
                to_play = 1 - dealer
            else:
                to_play = dealer

            while not agreed:
                player = players[to_play]
                assert player.stack >= 0, player.stack
                assert not((player.stack == 0) and not player.is_all_in), (player, player.is_all_in, actions)
                action = player.play(board, pot, actions, b_round, players[1 - to_play].stack, players[1 - to_play].side_pot, BLINDS)
                if action.type == 'null':
                    to_play = 1 - to_play
                    null += 1
                    if null >= 2:
                        agreed = True
                        break
                    continue
                if action.type in {'all in', 'bet', 'call'}:  # impossible to bet/call/all in 0
                    try:
                        assert action.value > 0
                    except AssertionError:
                        actions
                        raise AssertionError

                if action.type == 'call':
                    # if you call, it must be exactly the value of the previous bet or raise or all-in
                    if b_round > 0:
                        assert action.value + player.side_pot == players[1-player.id].side_pot, (action.value, actions[b_round][1-player.id][-1].value)
                    else:
                        if len(actions[b_round][1-player.id]) == 0:
                            assert action.value == 1
                        else:
                            assert action.value + player.side_pot == players[1-player.id].side_pot, (actions, action.value, actions[b_round][1 - player.id][-1].value)

                # RL : Store transitions in memory. Just for the agent
                if player.id == 0:
                    update_memory(MEMORY, players, action, new_game, board, pot, dealer, actions)
                ##############

                if action.type == 'raise':  # a raise is defined by the amount above the opponent side pot
                    value = action.value + players[1-to_play].side_pot - player.side_pot - (len(actions[0][player.id])==0)*(b_round==0)*BLINDS[1-player.is_dealer]
                else:
                    value = action.value
                player.side_pot += value
                player.stack -= value
                assert player.stack >= 0, (player.stack, actions, action, value)
                pot += value
                assert pot + players[0].stack + players[1].stack == 2*INITIAL_MONEY, (players, actions, action)
                assert not ((player.stack == 0) and action.type != 'all in'), (actions, action, player)
                actions[b_round][player.id].append(action)

                if action.type == 'all in':
                    all_in += 1
                    player.is_all_in = True
                    if action.value <= players[1-to_play].side_pot:
                        # in this case, the all in is a call and it leads to showdown
                        all_in += 1
                elif (action.type == 'call') and (all_in == 1):
                    # in this case, you call a all-in and it goes to showdown
                    all_in += 1

                # break if fold
                if action.type == 'fold':
                    fold_occured = True
                    players[0].contribution_in_this_pot = players[0].side_pot*1
                    players[1].contribution_in_this_pot = players[1].side_pot*1
                    players[0].side_pot = 0
                    players[1].side_pot = 0
                    winner = 1 - to_play
                    if verbose:
                        print(players[winner].name + ' wins because its opponent folded')
                    break

                # decide whether it is the end of the betting round or not
                agreed = agreement(actions, b_round)
                to_play = 1 - to_play

            players[0].contribution_in_this_pot += players[0].side_pot * 1
            players[1].contribution_in_this_pot += players[1].side_pot * 1
            players[0].side_pot = 0
            players[1].side_pot = 0

            # potentially stop the episode
            if fold_occured:
                break
        else:
            # deal all remaining cards
            for j in range(b_round, 4):
                deal(deck, players, board, j, verbose=verbose)

            # keep track of new state
            state_ = [cards_to_array(players[0].cards), cards_to_array(board), pot, players[0].stack, players[1].stack,
                      np.array(BLINDS), dealer, actions_to_array(actions)]
            MEMORY[-1]["s'"] = state_

            # end the episode
            players[0].contribution_in_this_pot += players[0].side_pot * 1
            players[1].contribution_in_this_pot += players[1].side_pot * 1
            players[0].side_pot = 0
            players[1].side_pot = 0
            break

    # winner gets money and variables are updated
    split = False
    if not fold_occured:
        hand_1 = evaluate_hand(players[1].cards+board)
        hand_0 = evaluate_hand(players[0].cards+board)

        # possible split
        if hand_1[1] == hand_0[1]:
            if hand_1[2] == hand_0[2]:
                split = True
            else:
                for card_0, card_1 in zip(hand_0[2], hand_1[2]):
                    if card_0 < card_1:
                        winner = 1
                        break
                    elif card_0 == card_1:
                        continue
                    else:
                        winner = 0
                        break

        # no split
        else:
            winner = int(hand_1[1] > hand_0[1])

        if verbose:
            if not split:
                print(players[0].name + ' cards : ' + str(players[0].cards) + ' and score: ' + str(hand_0[0]))
                print(players[1].name + ' cards : ' + str(players[1].cards) + ' and score: ' + str(hand_1[0]))
                print(players[winner].name + ' wins')
            else:
                print(players[0].name + ' cards : ' + str(players[0].cards) + ' and score: ' + str(hand_0[0]))
                print(players[1].name + ' cards : ' + str(players[1].cards) + ' and score: ' + str(hand_1[0]))
                print('Pot split')
    if not split:
        # if the winner isn't all in, it takes everything
        if players[winner].stack > 0:
            players[winner].stack += pot

        # if the winner is all in, it takes only min(what it put in the pot*2, pot)
        else:
            s_pot = players[0].contribution_in_this_pot, players[1].contribution_in_this_pot
            if s_pot[winner]*2 > pot:
                players[winner].stack += pot
            else:
                players[winner].stack += 2*s_pot[winner]
                players[1 - winner].stack += pot - 2*s_pot[winner]

        ##### RL #####
        # If the agent won, gives it the chips
        if winner == 0:
            # if the opponent immediately folds, then the MEMORY is empty and there is no reward to add since you didn't have the chance to act
            if len(MEMORY) > 0:
                MEMORY[-1]['r'] += pot
        ##############
    else:
        pot_0, pot_1 = players[0].contribution_in_this_pot, players[1].contribution_in_this_pot
        assert pot_0 + pot_1 + players[0].stack + players[1].stack == 2*INITIAL_MONEY, (pot_0, pot_1, players[0].stack, players[1].stack)
        players[0].stack += pot_0
        players[1].stack += pot_1
        players[0].contribution_in_this_pot = 0
        players[1].contribution_in_this_pot = 0
        ##### RL #####
        MEMORY[-1]['r'] += pot_0
        split = False
        ##############

    pot = 0
    dealer = 1 - dealer
    players[dealer].is_dealer = True
    players[1 - dealer].is_dealer = False
    players[0].cards = []
    players[1].cards = []
    players[0].contribution_in_this_pot = 0
    players[1].contribution_in_this_pot = 0
    assert players[1].side_pot == players[0].side_pot == 0

    # is the game finished ?
    if players[0].stack == 0 or players[1].stack == 0:
        new_game = True
    else:
        new_game = False

    episodes += 1

    # @todo: train Q network here
