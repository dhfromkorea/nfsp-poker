"""
Define classes for :
    - Card
    - Deck (a set of cards)
    - Player (two cards, a stack, dealer or not, all in or not)
    - Action (the bets)

Setting the blinds, the dealer button, deal cards, processing actions, transforming the board into a
"""
from itertools import product
from sklearn.utils import shuffle
from game.config import BLINDS
from game.utils import sample_categorical, variable
import numpy as np
import torch as t


class Card:
    RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
    SUITS = ['h', 'c', 's', 'd']
    IDX_TO_RANK = {k:v for k,v in enumerate(RANKS)}
    RANK_TO_IDX = {v:k for k,v in enumerate(RANKS)}
    IDX_TO_SUIT = {k:v for k,v in enumerate(SUITS)}

    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit

    @property
    def value(self):
        """The index of the card from 1 to 13"""
        return Card.RANKS.index(self.rank) + 1

    def __repr__(self):
        return self.rank + self.suit

    def __eq__(self, other):
        return (self.rank == other.rank) and (self.suit == other.suit)

    def __le__(self, other):
        return self.RANK_TO_IDX[self.rank] <= self.RANK_TO_IDX[other.rank]

    def __ge__(self, other):
        return self.RANK_TO_IDX[self.rank] >= self.RANK_TO_IDX[other.rank]

    def __lt__(self, other):
        return self.RANK_TO_IDX[self.rank] < self.RANK_TO_IDX[other.rank]

    def __gt__(self, other):
        return self.RANK_TO_IDX[self.rank] > self.RANK_TO_IDX[other.rank]


class Deck:
    """A set of cards"""
    def __init__(self):
        self.cards = []

    def populate(self):
        cards_tuples = list(product(Card.RANKS, Card.SUITS))
        self.cards = [Card(t[0], t[1]) for t in cards_tuples]

    def shuffle(self):
        self.cards = shuffle(self.cards)


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
    else:
        # at preflop, if the SB called and the BB raised, it continues playing. If the SB raised and the BB called, it ends the preflop
        if len(actions[0][0]) == len(actions[0][1]) == 1:
            if actions[betting_round][dealer][-1].type == 'raise' and actions[betting_round][1-dealer][-1].type == 'call':
                return True
        # at preflop, if the BB raised the call of the SB, then there is no more special situation
        else:
            if (actions[betting_round][0][-1].type == 'raise' and actions[betting_round][1][-1].type == 'call') or (actions[betting_round][0][-1].type == 'call' and actions[betting_round][1][-1].type == 'raise'):
                return True
    # 1 fold
    if actions[betting_round][0][-1].type == 'fold' or actions[betting_round][1][-1].type == 'fold':
        return True
    # 1 check 1 call
    if (actions[betting_round][0][-1].type == 'check' and actions[betting_round][1][-1].type == 'call') or (actions[betting_round][0][-1].type == 'call' and actions[betting_round][1][-1].type == 'check'):
        return True
    # 1 check 1 bet
    if (actions[betting_round][0][-1].type == 'check' and actions[betting_round][1][-1].type == 'bet') or (actions[betting_round][0][-1].type == 'bet' and actions[betting_round][1][-1].type == 'check'):
        return False
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
        # SB.contribution_in_this_pot += BLINDS[0]
        SB.stack -= BLINDS[0]
        sb_paid = BLINDS[0]
    else:
        SB.side_pot += SB.stack
        # SB.contribution_in_this_pot += SB.stack
        sb_paid = SB.stack
        SB.stack = 0
    if BB.stack >= BLINDS[1]:
        BB.side_pot += BLINDS[1]
        bb_paid = BLINDS[1]
        # BB.contribution_in_this_pot += BLINDS[1]
        BB.stack -= BLINDS[1]
    else:
        BB.side_pot += BB.stack
        # BB.contribution_in_this_pot += BB.stack
        bb_paid = BB.stack
        BB.stack = 0

    if verbose:
        print('\n'+ SB.name + ' (' + str(SB.stack) + ') '+ ' paid the small blind ')
        print(BB.name + ' (' + str(BB.stack) + ') '+ ' paid the big blind ')
    return sb_paid + bb_paid


def set_dealer(players, verbose=False):
    """Randomly set the dealer"""
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


def cards_to_array(cards):
    """
    Convert a list of cards (the board or the hand) into a numpy array to be passed as input of the DQN
    :param cards: a list of Card objects
    :return: an array representing these cards.
             Note that if there are more than 2 cards (i.e if this is the board),
             then the first 3 card are grouped together
    """
    if len(cards) == 2:
        array = np.zeros((13, 4))
        for card in cards:
            value = Card.RANKS.index(card.rank)
            suit = Card.SUITS.index(card.suit)
            array[value, suit] = 1
        return array
    elif len(cards) == 1 or len(cards) > 5:
        raise ValueError('there should be either 0, 2,3,4, or 5 cards')
    elif len(cards) == 0:
        return np.zeros((3, 13, 4))
    elif len(cards) == 3:
        array = np.zeros((3, 13, 4))
        for card in cards:
            value = Card.RANKS.index(card.rank)
            suit = Card.SUITS.index(card.suit)
            array[0, value, suit] = 1
        return array
    elif len(cards) == 4:
        array = np.zeros((3, 13, 4))
        for i, card in enumerate(cards):
            value = Card.RANKS.index(card.rank)
            suit = Card.SUITS.index(card.suit)
            array[int(i >= 3), value, suit] = 1
        return array
    elif len(cards) == 5:
        array = np.zeros((3, 13, 4))
        for i, card in enumerate(cards):
            value = Card.RANKS.index(card.rank)
            suit = Card.SUITS.index(card.suit)
            if i < 3:
                idx = 0
            elif i == 3:
                idx = 1
            elif i == 4:
                idx = 2
            array[idx, value, suit] = 1
        return array


def array_to_cards(array):
    if array.max() == 0:
        return []
    if len(array.shape) == 2:
        i, j = np.nonzero(array)
    elif len(array.shape) == 3:
        i, j = np.nonzero(array.sum(0))
    elif len(array.shape) == 4:
        i, j = np.nonzero(array.sum((0, 1)))
    return [Card(Card.IDX_TO_RANK[ii], Card.IDX_TO_SUIT[jj]) for ii, jj in zip(i, j)]


class Action:
    """The possible types of actions"""

    BET_BUCKETS = {
        -1: (None, None),  # this is fold
        0: (0, 0),  # this is check
        1: (1, 1),
        2: (2, 2),
        3: (3, 4),
        4: (5, 6),
        5: (7, 10),
        6: (11, 14),
        7: (15, 19),
        8: (20, 25),
        9: (26, 30),
        10: (31, 40),
        11: (41, 60),
        12: (61, 80),
        13: (81, 100),
        # 14: (101, 200)  # useless ?
    }

    def __init__(self, type, value=0, min_raise=None, total=None):
        assert type in {'call', 'check', 'all in', 'fold', 'raise', 'bet', 'null'}
        self.type = type
        self.value = value
        self.min_raise = min_raise
        self.total = value if total is None else total

    def __repr__(self):
        return self.type + ' ' + str(self.value)

    def __eq__(self, other):
        return (self.type == other.type) and (self.value == other.value) and (self.min_raise == other.min_raise)


def idx_to_bucket(idx):
    """Mapping between the indexes of the Q values (numpy array) and the idx of the actions in Action.BET_BUCKET"""
    if idx <= 1:
        return idx - 1
    else:
        return idx


def bucket_to_action(bucket, actions, b_round, player, opponent_side_pot):
    """
    Actions are identified by discrete buckets (see Action.BET_BUCKETS)
    We need to choose an action from a given bucket
    For this, if the bet is in a given bucket, we choose the minimum bet that allows to be in this bucket

    The bucket is supposed to be chosen by `authorized_actions_bucket` first
    :param opponent_side_pot:
    :param bucket: the id of the bucket
    :param actions: a dict representing the actions that were taken in this episode {b_round: {player: [actions]}}
    :param b_round: the idx of the betting round (0: preflop, 1: flop, ...)
    :param player: the player who is playing
    :return: an Action object
    """
    # there are some simple case
    if bucket == 14:
        return Action('all in', player.stack)
    elif bucket == 0:
        return Action('check')
    elif bucket == -1:
        return Action('fold')

    # the other cases can fall in different categories: bet/call/raise
    else:
        if opponent_side_pot == 0:
            # this is a bet because the opponent didn't play yet or checked
            return Action('bet', Action.BET_BUCKETS[bucket][0])
        else:
            if get_call_bucket(opponent_side_pot - player.side_pot) == bucket:
                # this is a call because the bucket contains the value of the side pot of the opponent
                value_to_bet = opponent_side_pot - player.side_pot
                return Action('call', value=value_to_bet)
            elif get_call_bucket(opponent_side_pot - player.side_pot) < bucket:  # if the opponent SP is 2 and yours is 1, the call bucket is 1
                # this is a raise
                # it can be a min-raise
                try:
                    opponent_last_bet_or_raise = actions[b_round][1-player.id][-1].value
                except IndexError:
                    opponent_last_bet_or_raise = 0
                raise_value = get_raise_from_bucket(bucket, actions, b_round, player, opponent_side_pot, raise_val=opponent_last_bet_or_raise)
                try:
                    if raise_value == actions[b_round][1-player.id][-1].value:
                        if raise_value + opponent_side_pot - player.side_pot == player.stack:
                            return Action('all in', value=player.stack)
                        elif raise_value + opponent_side_pot - player.side_pot < player.stack:
                            return Action('raise', value=raise_value, min_raise=True, total=raise_value + opponent_side_pot - player.side_pot)
                        else:
                            raise ValueError(('It should\'nt happen', bucket, actions, raise_value, opponent_side_pot, player.stack, player.side_pot))
                    else:
                        if raise_value + opponent_side_pot - player.side_pot == player.stack:
                            return Action('all in', value=player.stack)
                        elif raise_value + opponent_side_pot - player.side_pot < player.stack:
                            return Action('raise', value=raise_value, min_raise=False, total=raise_value + opponent_side_pot - player.side_pot)
                        else:
                            raise ValueError(('It should\'nt happen', bucket, actions, raise_value, opponent_side_pot, player.stack, player.side_pot))
                except IndexError:  # in this case, you are small blind and raise the BB
                    assert b_round == 0
                    assert player.is_dealer
                    assert len(actions[0][1-player.id]) == 0
                    if raise_value == 2:
                        if raise_value + opponent_side_pot - player.side_pot == player.stack:
                            return Action('all in', value=player.stack)
                        elif raise_value + opponent_side_pot - player.side_pot < player.stack:
                            return Action('raise', value=raise_value, min_raise=True, total=raise_value + opponent_side_pot - player.side_pot)
                        else:
                            raise ValueError(('It should\'nt happen', bucket, actions, raise_value, opponent_side_pot, player.stack, player.side_pot))
                    else:
                        if raise_value + opponent_side_pot - player.side_pot == player.stack:
                            return Action('all in', value=player.stack)
                        elif raise_value + opponent_side_pot - player.side_pot < player.stack:
                            return Action('raise', value=raise_value, min_raise=False, total=raise_value + opponent_side_pot - player.side_pot)
                        else:
                            raise ValueError(('It should\'nt happen', bucket, actions, raise_value, opponent_side_pot, player.stack, player.side_pot))
    raise ValueError((actions, player, bucket, b_round))

def get_raise_from_bucket(bucket, actions, b_round, player, opponent_side_pot, raise_val=0):
    """
    Note that the raise is what you BET ABOVE THE CURRENT SIDE POT OF THE OPPONENT (TAKING INTO ACCOUNT ALL OF ITS BETS IN THE CURRENT ROUND)
    :param bucket:
    :param actions:
    :param b_round:
    :param player:
    :param opponent_side_pot:
    :return:
    """
    min_range_of_your_bucket, max_range_of_your_bucket = Action.BET_BUCKETS[bucket]

    min_side_pot_to_match_for_raise_bucket, min_side_pot_to_match_for_raise = get_min_raise_bucket(opponent_side_pot, actions, b_round, player, raise_val=raise_val, return_min_raise=True)
    if max_range_of_your_bucket + player.side_pot < min_side_pot_to_match_for_raise:
        raise ValueError(('This is not a raise', player, actions, opponent_side_pot, min_side_pot_to_match_for_raise, min_range_of_your_bucket, max_range_of_your_bucket))
    if min_range_of_your_bucket + player.side_pot > min_side_pot_to_match_for_raise:
        try:
            assert min_range_of_your_bucket >= actions[b_round][1-player.id][-1].value, (actions, min_side_pot_to_match_for_raise, min_range_of_your_bucket, opponent_side_pot, actions[b_round][1-player.id][-1].value)
        except IndexError:
            assert min_range_of_your_bucket > 1
        return min_range_of_your_bucket + player.side_pot - opponent_side_pot
    else:
        return min_side_pot_to_match_for_raise - opponent_side_pot


def sample_action(idx, probabilities):
    """
    Sample from categorical distribution
    """
    try:
        return idx[sample_categorical(probabilities)]
    except:
        raise ValueError(probabilities)


def get_call_bucket(bet):
    """Returns the bucket that contains `bet`"""
    for bucket, range in Action.BET_BUCKETS.items():
        if bucket == -1:
            continue
        if range[0] <= bet <= range[1] <= 100:
            return bucket
    return 14


def get_max_bet_bucket(stack):
    """Returns the biggest bucket you can use to make a bet. Note that it is below the one that leads you to all-in"""
    assert 0 < stack <= 200, stack
    if stack == 1:  # you can just go all-in
        return 1
    for bucket, range in Action.BET_BUCKETS.items():
        if bucket == -1:
            continue
        if range[0] <= stack <= range[1]:
            return bucket
    return 13


def get_min_raise_bucket(opponent_side_pot, actions, b_round, player, raise_val=0, return_min_raise=False):
    """
    Gives you the bucket that contains the min raise you can do
    Note that, to decrease the dimensionality of the inputs, only 2 min-raises are allowed. That way, the total number of actions
    per betting round is kept small (6 max, corresponding to check/bet/2 min raises/and x2 raises at least by the RL agent)
    """
    actions_you_took = actions[b_round][player.id]
    n_min_raise = sum([a.min_raise for a in actions_you_took if a.type == 'raise'])
    if n_min_raise >= 2:
        # now you no longer have right to min raise
        minimum_side_pot_you_have_to_match = 2*opponent_side_pot
        if not return_min_raise:
            return get_call_bucket(minimum_side_pot_you_have_to_match - player.side_pot)
        else:
            return get_call_bucket(minimum_side_pot_you_have_to_match - player.side_pot), minimum_side_pot_you_have_to_match
    else:
        # you have right to min-raise
        # if you are small blind, your first raise is at least 2 (i.e bet 3 since you already have the small blind, i.e bucket 3)
        if player.is_dealer and b_round == 0 and len(actions[0][1-player.id]) == 0:
            if not return_min_raise:
                return 3
            else:
                return 3, 4  # bucket 3, 4 in total

        minimum_side_pot_you_have_to_match = raise_val + opponent_side_pot
        if not return_min_raise:
            return get_call_bucket(minimum_side_pot_you_have_to_match - player.side_pot)
        else:
            return get_call_bucket(minimum_side_pot_you_have_to_match - player.side_pot), minimum_side_pot_you_have_to_match


def authorized_actions_buckets(player, actions, b_round, opponent_side_pot):
    """
    Gives you the buckets you have right to choose
    :param player:
    :param actions:
    :param b_round:
    :param opponent_side_pot:
    :return:
    """
    opponent_id = 1 - player.id

    # in this case, the opponent already played before you in this betting round
    try:
        last_action_taken_by_opponent = actions[b_round][opponent_id][-1]

        # what if it checked ?
        if last_action_taken_by_opponent.type == 'check':
            # you cannot fold
            check_bucket = 0
            min_bet_bucket = 2
            max_bet_bucket = get_max_bet_bucket(player.stack)
            assert max_bet_bucket != 14
            # bucket_to_action(call_bucket, actions, b_round, player, opponent_side_pot).total >= player.stack
            if min_bet_bucket < max_bet_bucket or ((min_bet_bucket == max_bet_bucket) and (Action.BET_BUCKETS[min_bet_bucket][1] > Action.BET_BUCKETS[min_bet_bucket][0])):
                if bucket_to_action(max_bet_bucket, actions, b_round, player, opponent_side_pot).total >= player.stack:
                    return [check_bucket] + list(range(min_bet_bucket, max_bet_bucket)) + [14]
                else:
                    return [check_bucket] + list(range(min_bet_bucket, max_bet_bucket+1)) + [14]
            else:
                return [check_bucket] + [14]

        # what if it called?
        elif last_action_taken_by_opponent.type == 'call':
            if b_round > 0:
                raise ValueError('This case shouldn\'t happen because a call should lead to the next betting round')

            # for preflop you can raise the call of the small blind
            else:
                assert not player.is_dealer, (actions, player.id, player.is_dealer)
                assert len(actions[player.id][0]) == 0, (actions, player.id, player.is_dealer)
                check_bucket = 0
                max_raise_bucket = get_max_bet_bucket(player.stack)
                if player.stack <= 2:
                    return [check_bucket, 14]
                else:
                    assert max_raise_bucket != 14
                    return [check_bucket] + list(range(2, max_raise_bucket+1)) + [14]

        # what if it bet ?
        elif last_action_taken_by_opponent.type == 'bet':
            call_bucket = get_call_bucket(last_action_taken_by_opponent.value)
            assert opponent_side_pot == last_action_taken_by_opponent.value
            min_raise_bucket = get_min_raise_bucket(last_action_taken_by_opponent.value, actions, b_round, player, raise_val=last_action_taken_by_opponent.value)
            max_raise_bucket = get_max_bet_bucket(player.stack)
            assert min_raise_bucket > call_bucket, 'buckets are not well calibrated: a bet and a raise can be in the same bucket'
            if max_raise_bucket <= call_bucket:
                # in this case, all your money is below the bet of the opponent, so you can only fold or go all-in
                return [-1, 14]
            else:
                if opponent_side_pot < player.side_pot + player.stack:
                    if min_raise_bucket == 14:
                        return [-1] + [call_bucket] + [14]
                    min_raise_val = get_raise_from_bucket(min_raise_bucket, actions, b_round, player, opponent_side_pot, raise_val=last_action_taken_by_opponent.value)
                    try:
                        max_raise_val = get_raise_from_bucket(max_raise_bucket, actions, b_round, player, opponent_side_pot, raise_val=last_action_taken_by_opponent.value)
                    except ValueError:
                        max_raise_val = min_raise_val
                    if min_raise_val + opponent_side_pot > player.stack:
                        return [-1] + [call_bucket] + [14]
                    else:
                        if max_raise_val + opponent_side_pot > player.stack:
                            return [-1] + [call_bucket] + list(range(min_raise_bucket, max_raise_bucket)) + [14]
                        else:
                            assert max_raise_bucket != 14
                            return [-1] + [call_bucket] + list(range(min_raise_bucket, max_raise_bucket+1)) + [14]
                else:  # in this case your call is actually a all-in
                    return [-1] + list(range(min_raise_bucket, max_raise_bucket+1)) + [14]

        # what if it raised ?
        elif last_action_taken_by_opponent.type == 'raise':
            # you have right to do at most 2 min-raises (simplification), and then you have to double at least
            call_bucket = get_call_bucket(opponent_side_pot - player.side_pot)
            min_raise_bucket = get_min_raise_bucket(opponent_side_pot, actions, b_round, player, raise_val=last_action_taken_by_opponent.value)
            max_raise_bucket = get_max_bet_bucket(player.stack)

            assert min_raise_bucket > call_bucket, 'buckets are not well calibrated: a bet and a raise can be in the same bucket'
            if max_raise_bucket <= call_bucket or min_raise_bucket == 14:
                return [-1, 14]
            else:
                min_raise_val = get_raise_from_bucket(min_raise_bucket, actions, b_round, player, opponent_side_pot, raise_val=actions[b_round][1 - player.id][-1].value)
                if player.stack > - player.side_pot + opponent_side_pot + min_raise_val:
                    return [-1] + [call_bucket] + list(range(min_raise_bucket, max_raise_bucket+1)) + [14]
                else:
                    return [-1] + [call_bucket] + [14]

        # what if it is all-in ?
        elif last_action_taken_by_opponent.type == 'all in':
            call_bucket = get_call_bucket(opponent_side_pot - player.side_pot)
            max_bet_bucket = get_max_bet_bucket(player.stack)
            if max_bet_bucket <= call_bucket:
                if bucket_to_action(call_bucket, actions, b_round, player, opponent_side_pot).total >= player.stack:
                    return [-1, 14]
                else:
                    return [-1, call_bucket]
            else:
                return [-1, call_bucket]

    # in this case, you're first to play and can do basically whatever you want except folding and betting less than the big blind
    except IndexError:
        max_bet_bucket = get_max_bet_bucket(player.stack)
        if b_round == 0:  # preflop, you are SB and it is your first action. You can either fold, call, or raise at least 2 (i.e bet 3, i.e bucket 3)
            if max_bet_bucket == 1:
                return [-1, 14]
            else:
                assert max_bet_bucket != 14, (player, actions, b_round, opponent_side_pot)
                return [-1, 1] + list(range(3, max_bet_bucket+1)) + [14]
        else:  # after preflop, you are SB, it is your first move. You can either check or bet at least the BB
            if player.stack > 2:  # if you have enough money
                if Action.BET_BUCKETS[max_bet_bucket][0] < player.stack:
                    return [0] + list(range(2, max_bet_bucket+1)) + [14]
                else:
                    return [0] + list(range(2, max_bet_bucket)) + [14]  # in that case betting in max_bet_bucket is an all-in and not a bet
            else:  # if you have at most a BB
                return [0, 14]


def action_to_array(action):
    """
    Convert an action into a numpy array
    Actions will be `check`, `bet`, `call`, `raise`, `all in`
        `check` has to be included to be differentiated from not having played. If you just say that it is equivalent as
        betting 0, then you don't know whether it played or not
        `fold` should not be taken into account since it ends the game  @todo: sure of that ???
    :param action: an Action object
    :return: a numpy array
    """
    array = np.zeros((6,))
    if action.type == 'check':
        array[0] = 1
    elif action.type == 'bet':
        array[1] = action.value
    elif action.type == 'call':
        array[2] = action.value
    elif action.type == 'raise':
        array[3] = action.total  # action.value is the value of the raise, not the total value of the bet itself
    elif action.type == 'all in':
        array[4] = action.value
    elif action.type == 'fold':
        array[5] = 1
    return array


def actions_to_array(actions):
    """
    Convert a sequence of actions into several numpy arrays
    :param actions: a dict {b_round: {player: [actions]}}
    :return: 4 arrays, one per betting round
    """
    all_plays = []
    for b_round, players in actions.items():
        if b_round == -1:
            continue
        b_round_plays = np.zeros((6, 5, 2))  # 6: max number of actions in one round. 5: total number of possible actions. 2: number of players. 0 is the agent and 1 its opponent
        for player, plays in players.items():
            for k, action in enumerate(plays):
                try:
                    b_round_plays[k, :, player] = action_to_array(action)[:-1]  # if a folds happen, it ends the game, so that it is not needed to take a decision in a given betting round (if a fold had occured, you wouldnt have to take a decision)
                except IndexError:
                    raise IndexError(k, actions, action, b_round)
        all_plays.append(b_round_plays)
    return all_plays

def bucket_encode_actions(actions, cuda=False):
    """
    NOTE THAT THIS IS NOT ONE HOT ENCODING
    THIS IS RATHER BUCKET ENCODING (PUT AN ACTION REPRESENTED AS A 6x1 ARRAY TO AN INTEGER BUCKET)
    IT CAN BE USED IN THE LOSS

    YOU SHOULD ADD +1 (see the keys of `Action.BET_BUCKETS` to understand why)

    :param actions: a VARIABLE of size batch_size x 5 (il y a 5 types d'actions: check, bet, call, raise, all-in)
    :return: a VARIABLE of size batch_size x 14 (il y a 14 buckets)
    """
    values, indices = t.max(actions, -1)
    actions_buckets = variable(np.zeros(values.data.cpu().numpy().shape), cuda=cuda)
    actions_buckets[indices==0] = 0  # check
    actions_buckets[indices==4] = 14  # all in
    actions_buckets[indices==5] = -1  # fold
    for bucket_idx in range(1, 14):
        indicator = lambda x: bucket_idx*(x>=Action.BET_BUCKETS[bucket_idx][0]).float()*((x<=Action.BET_BUCKETS[bucket_idx][1]).float())
        mask = (indices != 0)*(indices != 5)*(indices != 4)
        actions_buckets[mask] += indicator(values[mask])
    return actions_buckets
