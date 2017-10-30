from itertools import product
from sklearn.utils import shuffle
from collections import Counter

BLINDS = (10, 20)


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

    def cash(self, v):
        self.stack = v

    def play(self, board, pot, actions, b_round, opponent_stack):
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
        action = self.strategy(self.id, self.is_dealer, self.cards, board, pot, actions, b_round, opponent_stack,
                               verbose=self.verbose, player_name=self.name)
        self.stack -= action.value
        return action

    def __repr__(self):
        if self.name is None:
            name = str(self.id)
        else:
            name = self.name
        if self.is_dealer:
            addon = 'D\n' + str(self.stack) + '\n'
        else:
            addon = str(self.stack) + '\n'
        return name + '\n' + addon + (' '.join([str(c) for c in self.cards]))


class Action:
    """The possible types of actions"""
    def __init__(self, type, value=0):
        assert type in {'call', 'all in', 'fold', 'raise', 'bet', 'null'}
        self.type = type
        self.value = value

    def __repr__(self):
        return self.type


def agreement(actions, betting_round):
    """
    Verify whether two players came to an agreement, meaning that the betting round can end now
    :param actions: a dict {betting_round: {player_id: [actions]}}
    :param betting_round: the id of the betting round (0:preflop, 1:flop,...)
    :return: True/False
    """
    if len(actions[betting_round][0]) == 0 or len(actions[betting_round][1]) == 0:
        return False
    # 1 fold
    if actions[betting_round][0][-1].type == 'fold' or actions[betting_round][1][-1].type == 'fold':
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
    SB = players[0] if not players[0].is_dealer else players[1]
    BB = players[0] if not players[1].is_dealer else players[1]
    if verbose:
        print(SB.name + ' paid the small blind ')
        print(BB.name + ' paid the big blind ')
    SB.stack -= BLINDS[0]
    BB.stack -= BLINDS[1]
    return sum(BLINDS)


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


names = {1: 'deuce',
         2: 'three',
         3: 'four',
         4: 'five',
         5: 'six',
         6: 'seven',
         7: 'eight',
         8: 'nine',
         9: 'ten',
         10: 'jack',
         11: 'queen',
         12: 'king',
         13: 'ace'}


def cn(value):
    return names[value]


def is_straight(values, length):
    hand = set(values)
    if 13 in hand:
        hand.add(0)

    for low in (10, 9, 8, 7, 6, 5, 4, 3, 2, 1):
        needed = set(range(low, low + length))
        if len(needed - hand) <= 0:
            return (low+length)-1
    return 0


# @todo: check that this works correctly
def evaluate_hand(cards):
    values = []
    raw_values = []
    suits = []
    flush = False
    high_card = True  # False if anything but a high card remains

    for card in cards:
        values.append(card.value)
        suits.append(card.suit)

    for v in values:
        raw_values.append(v)

    value_count = Counter(values)
    suit_count = Counter(suits)

    # put values in order of rank
    values.sort(reverse=True)

    # set up variables
    pair_l = []
    trip_l = []
    quad_l = []
    multiples_l = [0, 0, pair_l, trip_l, quad_l]  # 0,0 are dummies
    remove_list = []  # list of multiples to be removed
    winning_cards = []
    rep = ''
    hand_value = 0
    tie_break = 0

    limit = min(5, len(values))
    straight = is_straight(values, limit)

    for key, value in value_count.items():
        if value > 1:
            high_card = False
            multiples_l[value].append(key)
            for element in values:
                if element == key:
                    remove_list.append(element)
                    winning_cards.append(element)

            for item in remove_list:
                values.remove(item)

            winning_cards.sort(reverse=True)

            # used to determine ties between hands
            tie_break = values
            # clear the remove list for the next histogram iteration
            remove_list = []

    pair_l.sort(reverse=True)

    # avoid having three pairs
    if len(pair_l) == 3:
        tie_break.append(winning_cards[5:])

    for key, value in suit_count.items():
        flush_score = 0
        if value == 5:
            flush = True
            high_card = False
        else:
            flush_score = value

    if len(pair_l) == 1 and trip_l == []:
        rep = ('pair of ' + cn(pair_l[0]) + 's')
        hand_value = 100 + (sum(winning_cards[:2]))
        tie_break = values[:3]

    elif len(pair_l) > 1:
        rep = ('two pair -' + cn(pair_l[0]) + 's and ' + cn(pair_l[1]) + 's ')
        hand_value = 200 + (sum(winning_cards[:4]))
        tie_break = values[:1]

    elif trip_l and pair_l == []:
        rep = ('trip ' + cn(trip_l[0]) + 's ')
        hand_value = 300 + (sum(winning_cards[:3]))
        tie_break = values[:2]

    elif straight > 0 and not flush:
        rep = ('Straight, ' + cn(straight) + ' high')
        hand_value = 400 + straight

    elif flush:

        flush_l = []
        # find out the values of each flush card for comparison
        for card in cards:
            if key in card.suit:
                flush_l.append(card.value)
        flush_l.sort(reverse=True)
        rep = ('Flush, ' + cn(flush_l[0]) + ' high')
        hand_value = 500 + (int(flush_l[0]))
        tie_break = flush_l

    elif len(trip_l) == 1 and len(pair_l) >= 1:
        rep = ('full house - ' + cn(trip_l[0]) + 's full of ' + cn(pair_l[0]) + 's')
        hand_value = 600 + (sum(winning_cards[:3]))


    elif quad_l:
        rep = ('four ' + cn(quad_l[0]) + ' s')
        hand_value = 700 + (sum(winning_cards[:4]))
        tie_break = values[:1]

    elif (straight in range(1, 9)) and flush:
        rep = ('Straight flush, ' + cn(straight) + ' high')
        hand_value = 800 + straight

    else:
        rep = ('high card ' + cn(values[0]))
        hand_value = values[0]
        tie_break = values[:4]

    gappers = (raw_values[0]) - (raw_values[1])
    raw_data = (raw_values, flush_score, straight, gappers)

    return rep, hand_value, tie_break, raw_data


def compare_hands(players):
    return int(evaluate_hand(players[1].cards) > evaluate_hand(players[0].cards))
