"""
This script allows to run poker simulations while training an RL agent

Note:
To make the problem simpler, some actions are impossible
For example, we forbid that the agents min-raises more than twice in a given betting round.
It doesn't lose a lot of generality anyway, since it represents most situations. It has the advantage of greatly reducing the number of possible
actions per betting round

"""

from time import time
from evaluation import *
from game_utils import Deck, Player, set_dealer, blinds, deal, agreement, actions_to_array, action_to_array, cards_to_array
from q_network import *
from strategies import strategy_RL, strategy_random
from utils import *
from config import BLINDS
from experience_replay.experience_replay import ReplayBuffer


def update_memory(MEMORY, players, action, new_game, board, pot, dealer, actions):
    """
    Update the memory with transitions
    :param MEMORY: a list of dicts with keys `s`, `a`, `r`, `s'`
    :param players: a list of players
    :param action: an Action object
    :param new_game: whether it is a new game beginning (one of the player went bankrupt)
    :param board:
    :param pot: the amount of money in the pot
    :param dealer: who the dealer is (the id of the player)
    :param actions: a dict {b_round: {player: [actions_it_took]}}
    :return:
    """
    player = players[0]  # player 0 is the hero

    # matrify the interesting quantities
    state_ = [cards_to_array(player.cards), cards_to_array(board), pot, player.stack, players[1].stack,
              np.array(BLINDS), dealer, actions_to_array(actions)]
    action_ = action_to_array(action)
    reward_ = -action.value

    # store transition
    transition = {'s': state_, 'a': action_, 'r': reward_}
    if len(MEMORY) > 0 and not new_game:  # don't take into account transitions overlapping two different games
        # don't forget to store next state
        MEMORY[-1]["s'"] = state_
    MEMORY.append(transition)


verbose = True

# instantiate game
deck = Deck()
INITIAL_MONEY = 100*BLINDS[0]
Q = Q_network(10, 14)
players = [Player(0, strategy_RL(Q, True), INITIAL_MONEY, verbose=True, name='SB'),
           Player(1, strategy_RL(Q, True), INITIAL_MONEY, verbose=True, name='DH')]
# players = [Player(0, strategy_random, INITIAL_MONEY, verbose=True, name='SB'),
#            Player(1, strategy_random, INITIAL_MONEY, verbose=True, name='DH')]
board = []
dealer = set_dealer(players)
MEMORY = []  # a list of dicts with keys s,a,r,s'
new_game = True
episodes = 0
games = {'n': 0, '#episodes': []}  # some statistics on the games
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

    # PAY BLINDS
    pot = blinds(players, verbose=verbose)
    if verbose:
        print('pot: ' + str(pot))

    # SHUFFLE DECK AND CLEAR BOARD
    deck.populate()
    deck.shuffle()
    board = []
    players[0].is_all_in = False
    players[1].is_all_in = False

    # MONITOR ACTIONS
    # -1 is for the blinds
    actions = {b_round: {player: [] for player in range(2)} for b_round in range(-1, 4)}
    actions[-1][players[0].id] = players[0].side_pot
    actions[-1][players[1].id] = players[1].side_pot

    # dramatic events monitoring (fold, all-in)
    fold_occured = False
    null = 0  # once one or two all-ins occurred, actions are null. Count them to stop the loop
    all_in = 0  # 0, 1 or 2. If 2, the one of the player is all-in and the other is either all-in or called. In that case, things should be treated differently
    if players[0].stack == 0:  # in this case the blind puts the player all-in
        all_in = 2
        players[0].is_all_in = True
    if players[1].stack == 0:
        all_in = 2
        players[1].is_all_in = True

    # EPISODE ACTUALLY STARTS HERE
    for b_round in range(4):

        # DIFFERENTIATE THE CASES WHERE PLAYERS ARE ALL-IN FROM THE ONES WHERE NONE OF THEM IS
        if all_in < 2:
            # DEAL CARDS
            deal(deck, players, board, b_round, verbose=verbose)
            agreed = False  # True when the max bet has been called by everybody

            # PLAY
            if b_round != 0:
                to_play = 1 - dealer
            else:
                to_play = dealer

            while not agreed:

                # CHOOSE AN ACTION
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

                # RL : Store transitions in memory. Just for the agent
                if player.id == 0:
                    update_memory(MEMORY, players, action, new_game, board, pot, dealer, actions)

                # TRANSITION STATE DEPENDING ON THE ACTION YOU TOOK
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

                if action.type == 'raise':  # a raise is defined by the amount above the opponent side pot
                    value = action.value + players[1-to_play].side_pot - player.side_pot - (len(actions[0][player.id])==0)*(b_round==0)*BLINDS[1-player.is_dealer]
                else:
                    value = action.value

                # update pot
                player.side_pot += value
                player.stack -= value
                assert player.stack >= 0, (player.stack, actions, action, value)
                pot += value
                assert pot + players[0].stack + players[1].stack == 2*INITIAL_MONEY, (players, actions, action)
                assert not ((player.stack == 0) and action.type != 'all in'), (actions, action, player)
                actions[b_round][player.id].append(action)

                # DRAMATIC ACTION MONITORING
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

                # DECIDE WHETHER IT IS THE END OF THE BETTING ROUND OR NOT, AND GIVE LET THE NEXT PLAYER PLAY
                agreed = agreement(actions, b_round)
                to_play = 1 - to_play

            players[0].contribution_in_this_pot += players[0].side_pot * 1
            players[1].contribution_in_this_pot += players[1].side_pot * 1
            players[0].side_pot = 0
            players[1].side_pot = 0

            # POTENTIALLY STOP THE EPISODE IF FOLD OCCURRED
            if fold_occured:
                break
        else:
            # DEAL REMAINING CARDS
            for j in range(b_round, 4):
                deal(deck, players, board, j, verbose=verbose)

            # KEEP TRACK OF TRANSITIONS
            state_ = [cards_to_array(players[0].cards), cards_to_array(board), pot, players[0].stack, players[1].stack,
                      np.array(BLINDS), dealer, actions_to_array(actions)]
            MEMORY[-1]["s'"] = state_

            # END THE EPISODE
            players[0].contribution_in_this_pot += players[0].side_pot * 1
            players[1].contribution_in_this_pot += players[1].side_pot * 1
            players[0].side_pot = 0
            players[1].side_pot = 0
            break

    # WINNERS GETS THE MONEY.
    # WATCH OUT! TIES CAN OCCUR. IN THAT CASE, SPLIT
    split = False
    if not fold_occured:
        # compute the value of hands
        hand_1 = evaluate_hand(players[1].cards+board)
        hand_0 = evaluate_hand(players[0].cards+board)

        # decide whether to split or not
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

        # if no split, somebody won
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

        # RL
        # If the agent won, gives it the chips and reminds him that it won the chips
        if winner == 0:
            # if the opponent immediately folds, then the MEMORY is empty and there is no reward to add since you didn't have the chance to act
            if len(MEMORY) > 0:
                MEMORY[-1]['r'] += pot
    else:
        # SPLIT: everybody takes back its money
        pot_0, pot_1 = players[0].contribution_in_this_pot, players[1].contribution_in_this_pot
        assert pot_0 + pot_1 + players[0].stack + players[1].stack == 2*INITIAL_MONEY, (pot_0, pot_1, players[0].stack, players[1].stack)
        players[0].stack += pot_0
        players[1].stack += pot_1
        players[0].contribution_in_this_pot = 0
        players[1].contribution_in_this_pot = 0

        # RL : update the memory with the amount you won
        MEMORY[-1]['r'] += pot_0
        split = False

    # RESET VARIABLES
    pot = 0
    dealer = 1 - dealer
    players[dealer].is_dealer = True
    players[1 - dealer].is_dealer = False
    players[0].cards = []
    players[1].cards = []
    players[0].contribution_in_this_pot = 0
    players[1].contribution_in_this_pot = 0
    assert players[1].side_pot == players[0].side_pot == 0

    # IS IT THE END OF THE GAME ? (bankruptcy)
    if players[0].stack == 0 or players[1].stack == 0:
        new_game = True
    else:
        new_game = False

    episodes += 1

    # @todo: train Q network here
    # EXPERIENCE REPLAY SAMPLING
    # UPDATE Q WEIGHTS
    # UPDATE STRATEGY OF THE ADVERSARY
    # players[0].strategy = strategy_RL(Q, True)
    # Watch out, weights of the opponent should be kept frozen. Check that updating those of player doesn't help his
    # you may find the function Q.get_weights() and Q.set_weights(weights) useful. They are symmetrical of course
