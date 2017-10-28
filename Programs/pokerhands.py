# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 22:46:26 2017

@author: SrivatsanPC
"""

from collections import Counter
from operator import attrgetter
import pokerconst as pc


def cn(value):
    return pc.names[value]


def is_straight(values, length):
    hand = set(values)
    if 13 in hand:
        hand.add(0)

    for low in (10, 9, 8, 7, 6, 5, 4, 3, 2, 1):
        needed = set(range(low, low + length))
        if len(needed - hand) <= 0:
            return (low+length)-1  
    return 0


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
