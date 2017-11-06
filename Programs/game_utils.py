"""
Define classes for :
    - Card
    - Deck (a set of cards)
    - Player (two cards, a stack, dealer or not, all in or not)
    - Action (the bets)

Setting the blinds, the dealer button, deal cards

Evaluate what hand is better
"""
from itertools import product
from sklearn.utils import shuffle
from actions import Action

BLINDS = (1, 2)


class Card:
    RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
    SUITS = ['h', 'c', 's', 'd']

    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit

    @property
    def value(self):
        """The index of the card from 1 to 13"""
        return Card.RANKS.index(self.rank) + 1

    def __repr__(self):
        return self.rank + self.suit


class Deck:
    """A set of cards"""
    def __init__(self):
        self.cards = []

    def populate(self):
        cards_tuples = list(product(Card.RANKS, Card.SUITS))
        self.cards = [Card(t[0], t[1]) for t in cards_tuples]

    def shuffle(self):
        self.cards = shuffle(self.cards)


class Player:
    def __init__(self, id, strategy, stack, name=None, verbose=False):
        self.id = id
        self.cards = []
        self.stack = stack
        self.is_dealer = False
        self.is_all_in = False
        self.strategy = strategy
        self.verbose = verbose
        self.name = name
        self.side_pot = 0

    def cash(self, v):
        self.side_pot = 0
        self.stack = v

    def play(self, board, pot, actions, b_round, opponent_stack, opponent_side_pot, blinds=BLINDS):
        # if you are all in you cannot do anything
        if self.is_all_in:
            if self.verbose:
                print(self.name + ' did nothing (all in)')
            return Action('null')
        if b_round > 0:
            if actions[b_round - 1][0][-1].type == 'all in' or actions[b_round - 1][1][-1].type == 'all in':
                if self.verbose:
                    print(self.name + ' did nothing (all in)')
                return Action('null')

        action = self.strategy(self, board, pot, actions, b_round, opponent_stack, opponent_side_pot, blinds=blinds, verbose=self.verbose)
        try:
            if self.stack - action.value < 0:
                raise AttributeError
        except AttributeError:
            action = self.strategy(self, board, pot, actions, b_round, opponent_stack, opponent_side_pot, blinds=blinds, verbose=self.verbose)
            raise AttributeError((b_round, actions, action, self.id))
        return action

    def __repr__(self):
        if self.name is None:
            name = str(self.id)
        else:
            name = self.name
        if self.is_dealer:
            addon = 'D\n' + str(self.stack) + '\n' + str(self.side_pot) + '\n'
        else:
            addon = str(self.stack) + '\n' + str(self.side_pot) + '\n'
        return name + '\n' + addon + (' '.join([str(c) for c in self.cards]))


def agreement(actions, betting_round):
    """
    Verify whether two players came to an agreement, meaning that the betting round can end now
    :param actions: a dict {betting_round: {player_id: [actions]}}
    :param betting_round: the id of the betting round (0:preflop, 1:flop,...)
    :return: True/False
    """
    if len(actions[betting_round][0]) == 0 or len(actions[betting_round][1]) == 0:
        return False
    # 1 raise 1 call
    dealer = 0 if actions[-1][0] == 1 else 1
    if betting_round > 0:  # after preflop, any call/raise situation leads to the end of the betting round
        if (actions[betting_round][0][-1].type == 'raise' and actions[betting_round][1][-1].type == 'call') or (actions[betting_round][0][-1].type == 'call' and actions[betting_round][1][-1].type == 'raise'):
            return True
    else:  # at preflop, if the SB called and the BB raised, it continues playing. If the SB raised and the BB called, it ends the preflop
        if actions[betting_round][dealer][-1].type == 'raise' and actions[betting_round][1-dealer][-1].type == 'call':
            return True
    # 1 fold
    if actions[betting_round][0][-1].type == 'fold' or actions[betting_round][1][-1].type == 'fold':
        return True
    # 1 check 1 call
    if (actions[betting_round][0][-1].type == 'check' and actions[betting_round][1][-1].type == 'call') or (actions[betting_round][0][-1].type == 'call' and actions[betting_round][1][-1].type == 'check'):
        return True
    # 1 check 1 bet
    if (actions[betting_round][0][-1].type == 'check' and actions[betting_round][1][-1].type == 'bet') or (actions[betting_round][0][-1].type == 'bet' and actions[betting_round][1][-1].type == 'check'):
        return True
    # 2 checks
    if actions[betting_round][0][-1].type == 'check' and actions[betting_round][1][-1].type == 'check':
        return True
    # 1 bet 1 call
    if (actions[betting_round][0][-1].type == 'bet' and actions[betting_round][1][-1].type == 'call') or (
            actions[betting_round][0][-1].type == 'call' and actions[betting_round][1][-1].type == 'bet'):
        return True
    # 1 all-in and 1 call
    if (actions[betting_round][0][-1].type == 'all in' and actions[betting_round][1][-1].type == 'call') or (
                    actions[betting_round][0][-1].type == 'call' and actions[betting_round][1][-1].type == 'all in'):
        return True
    # 2 all-in
    if actions[betting_round][0][-1].type == 'all in' and actions[betting_round][1][-1].type == 'all in':
        return True
    return False


def blinds(players, verbose=False):
    """
    Players pay the blind
    :param players: a list of players
    :param verbose: whether to print stuff
    :return: the pot
    """
    SB = players[0] if players[0].is_dealer else players[1]
    BB = players[1 - SB.id]
    SB.side_pot = 0
    BB.side_pot = 0
    if SB.stack >= BLINDS[0]:
        SB.side_pot += BLINDS[0]
        SB.stack -= BLINDS[0]
        sb_paid = BLINDS[0]
    else:
        SB.side_pot += SB.stack
        sb_paid = SB.stack
        SB.stack = 0
    if BB.stack >= BLINDS[1]:
        BB.side_pot += BLINDS[1]
        bb_paid = BLINDS[1]
        BB.stack -= BLINDS[1]
    else:
        BB.side_pot += BB.stack
        bb_paid = BB.stack
        BB.stack = 0

    if verbose:
        print('\n'+ SB.name + ' (' + str(SB.stack) + ') '+ ' paid the small blind ')
        print(BB.name + ' (' + str(BB.stack) + ') '+ ' paid the big blind ')
    return sb_paid + bb_paid


def set_dealer(players, verbose=False):
    import random
    if random.random() > .5:
        if verbose:
            print(players[0].name + ' is dealer')
        players[0].is_dealer = True
        return 0
    else:
        if verbose:
            print(players[1].name + ' is dealer')
        players[1].is_dealer = True
        return 1


def deal(deck, players, board, b_round, verbose=False):
    """
    Deal the cards to player or board according to what betting round we are at
    :param b_round: the betting round
    :param players: a list of players
    :param verbose: True for printing stuffs
    """
    if b_round == 0:
        first_player = players[0] if players[0].is_dealer else players[1]
        second_player = players[(first_player.id + 1) % 2]
        first_player.cards.append(deck.cards.pop())
        second_player.cards.append(deck.cards.pop())
        first_player.cards.append(deck.cards.pop())
        second_player.cards.append(deck.cards.pop())
        if verbose:
            print(first_player.name + '\'s cards: ' + str(first_player.cards))
            print(second_player.name + '\'s cards: ' + str(second_player.cards))
    if b_round == 1:
        board.append(deck.cards.pop())
        board.append(deck.cards.pop())
        board.append(deck.cards.pop())
        if verbose:
            print('flop')
            print(board)
    if b_round == 2:
        board.append(deck.cards.pop())
        if verbose:
            print('turn')
            print(board)
    if b_round == 3:
        board.append(deck.cards.pop())
        if verbose:
            print('river')
            print(board)


def split_pot(actions, dealer, blinds=BLINDS):
    """
    Split the pot
    :param actions: a dict {b_round: {player: [actions]}}
    :return: pot_0, pot_1
    """
    pot = {0: 0, 1: 0}
    pot[0] += actions[-1][0]
    pot[1] += actions[-1][1]
    for b_round, players in actions.items():
        if b_round == -1:
            continue
        for player, actions in players.items():
            for action in actions:
                pot[player] += action.value
    return pot[0], pot[1]


if __name__ == '__main__':
    from strategies import strategy_limper
    INITIAL_MONEY = 100
    players = [Player(0, strategy_limper, 1, verbose=True, name='SB'),
               Player(1, strategy_limper, INITIAL_MONEY, verbose=True, name='DH')]
    players[1].is_dealer = True
    print(blinds(players))
