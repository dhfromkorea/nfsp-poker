from collections import Counter


def cn(value):
    return names[value]


def is_straight(values, length):
    hand = set(values)
    if 13 in hand:
        hand.add(0)

    for low in (10, 9, 8, 7, 6, 5, 4, 3, 2, 1):
        needed = set(range(low, low + length))
        if len(needed - hand) <= 0:
            return (low+length)-1, list(sorted(range(low, low+length), reverse=True))
    return 0, []


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
    straight, straight_cards = is_straight(values, limit)

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
            flush_suit = key
            flush = True
            high_card = False
        else:
            flush_score = value

    if len(pair_l) == 1:
        rep = ('pair of ' + cn(pair_l[0]) + 's')
        hand_value = 100 + (sum(winning_cards[:2]))
        tie_break = values[:3]

    if len(pair_l) > 1:
        rep = ('two pair -' + cn(pair_l[0]) + 's and ' + cn(pair_l[1]) + 's ')
        hand_value = 200 + (sum(winning_cards[:4]))
        tie_break = values[:1]

    if len(trip_l) == 1:
        rep = ('trip ' + cn(trip_l[0]) + 's ')
        hand_value = 300 + (sum(winning_cards[:3]))
        tie_break = values[:2]

    if straight > 0:
        rep = ('Straight, ' + cn(straight) + ' high')
        hand_value = 400 + straight
        tie_break = straight_cards

    if flush:
        flush_l = []
        # find out the values of each flush card for comparison
        for card in cards:
            if flush_suit in card.suit:
                flush_l.append(card.value)
        flush_l.sort(reverse=True)
        flush_l = flush_l[:5]
        rep = ('Flush, ' + cn(flush_l[0]) + ' high')
        hand_value = 500 + (int(flush_l[0]))
        tie_break = flush_l

    if len(trip_l) == 1 and len(pair_l) >= 1:
        rep = ('full house - ' + cn(trip_l[0]) + 's full of ' + cn(pair_l[0]) + 's')
        hand_value = 600 + (sum(winning_cards[:3]))
        tie_break = []

    if len(trip_l) == 2:
        highest = max(trip_l)
        lowest = min(trip_l)
        rep = ('full house - ' + cn(highest) + 's full of ' + cn(lowest) + 's')
        hand_value = 600 + (3*highest)
        tie_break = []

    if quad_l:
        rep = ('four ' + cn(quad_l[0]) + ' s')
        hand_value = 700 + (sum(winning_cards[:4]))
        tie_break = values[:1]

    if (straight in range(1, 9)) and flush:
        rep = ('Straight flush, ' + cn(straight) + ' high')
        hand_value = 800 + straight

    if hand_value == 0:
        rep = ('high card ' + cn(values[0]))
        hand_value = values[0]
        tie_break = values[:4]

    gappers = (raw_values[0]) - (raw_values[1])
    raw_data = (raw_values, flush_score, straight, gappers)

    return rep, hand_value, tie_break, raw_data


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


def compare_hands(players):
    return int(evaluate_hand(players[1].cards) > evaluate_hand(players[0].cards))