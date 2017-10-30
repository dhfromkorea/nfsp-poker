from utils import *
from strategies import strategy_limper

verbose = True

# instantiate game
deck = Deck()
players = [Player(0, strategy_limper, 1000, verbose=True, name='SB'), Player(1, strategy_limper, 1000, verbose=True, name='DH')]
board = []
dealer = set_dealer(players)

while True:
    # put blinds
    pot = blinds(players, verbose=verbose)

    # shuffle decks are clear board
    deck.populate()
    deck.shuffle()
    board = []

    # keep track of actions of each player for this episode
    actions = {b_round: {player: [] for player in range(2)} for b_round in range(4)}

    # fold checkbox
    fold_occured = False

    # betting rounds
    for b_round in range(4):
        # deal cards
        deal(deck, players, board, b_round, verbose=verbose)

        # play
        agreed = False
        if b_round != 0:
            to_play = (dealer + 1) % 2
        else:
            to_play = dealer
        while not agreed:
            player = players[to_play]
            action = player.play(board, pot, actions, b_round, players[(to_play + 1) % 2].stack)
            pot += action.value
            actions[b_round][player.id].append(action)

            # break if fold
            if action.type == 'fold':
                fold_occured = True
                winner = (to_play + 1) % 2
                break

            # decide if it is the end of the betting round
            agreed = agreement(actions, b_round)
            to_play = (to_play + 1) % 2

        if fold_occured:
            break

    # winner gets money and variables are updated
    if not fold_occured:
        winner = compare_hands(players)
    players[winner].stack += pot
    pot = 0
    dealer = (dealer + 1) % 2
    players[dealer].is_dealer = True
    players[1-dealer].is_dealer = False
