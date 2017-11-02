import numpy as np


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

    def __init__(self, type, value=0, min_raise=None):
        assert type in {'call', 'check', 'all in', 'fold', 'raise', 'bet', 'null'}
        self.type = type
        self.value = value
        self.min_raise = min_raise

    def __repr__(self):
        return self.type + ' ' + str(self.value)


def idx_to_bucket(idx):
    """Mapping between the indexes of the Q values (numpy array) and the idx of the actions in Action.BET_BUCKET"""
    return idx - 1


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
            if get_call_bucket(opponent_side_pot) == bucket:
                # this is a call because the bucket contains the value of the side pot of the opponent
                value_to_bet = opponent_side_pot - player.side_pot
                return Action('call', value=value_to_bet)
            elif get_call_bucket(opponent_side_pot) < bucket:
                # this is a raise
                # it can be a min-raise
                raise_value = get_raise_from_bucket(bucket, actions, b_round, player, opponent_side_pot)
                if raise_value == actions[b_round][1-player.id][-1].value:
                    return Action('raise', value=raise_value, min_raise=True)
                else:
                    return Action('raise', value=raise_value, min_raise=False)


def get_raise_from_bucket(bucket, actions, b_round, player, opponent_side_pot):
    """
    Note that the raise is what you BET ABOVE THE BET OF THE OPPONENT
    :param bucket:
    :param actions:
    :param b_round:
    :param player:
    :param opponent_side_pot:
    :return:
    """
    min_range, max_range = Action.BET_BUCKETS[bucket]
    if min_range <= opponent_side_pot + actions[b_round][1 - player.id][-1].value <= max_range:
        return actions[b_round][1 - player.id][-1].value
    else:
        return min_range - opponent_side_pot


def sample_action(idx, probabilities):
    """
    Sample from categorical distribution
    """
    return idx[np.random.multinomial(list(range(len(idx))), probabilities, size=1)]


def get_call_bucket(bet):
    """Returns the bucket that contains `bet`"""
    for bucket, range in Action.BET_BUCKETS.items():
        if range[0] <= bet <= range[1] < 100:
            return bucket
    return 14


def get_max_bet_bucket(stack):
    """Returns the biggest bucket you can use to make a bet. Note that it is below the one that leads you to all-in"""
    assert 0 < stack <= 200
    for bucket, range in Action.BET_BUCKETS.items():
        if range[0] <= stack <= range[1]:
            return bucket


def get_min_raise_bucket(opponent_side_pot, actions, b_round, player, raise_val=0):
    """
    Gives you the bucket that contains the min raise you can do
    Note that, to decrease the dimensionality of the inputs, only 2 min-raises are allowed. That way, the total number of actions
    per betting round is kept small (6 max, corresponding to check/bet/2 min raises/and x2 raises at least by the RL agent)
    """
    actions_you_took = actions[b_round][player.id]
    n_min_raise = sum([a for a in actions_you_took if a.type == 'raise'])
    if n_min_raise >= 2:
        # now you no longer have right to min raise
        min_raise = 2*(opponent_side_pot+raise_val)
        return get_call_bucket(min_raise)
    else:
        # you have right to min-raise
        min_raise = raise_val*2 + opponent_side_pot
        return get_call_bucket(min_raise)


def authorized_actions_buckets(player, actions, b_round, opponent_side_pot):
    """
    Gives you the buckets you have right to choose
    Note that the buckets returned correspond to the total side pot you have right to put on the table
    For example, if the opponent bet 10 and that you can raise 10, the min-raise bucket will be the 7th (16, 20)
    The actual raised value fixed in the `bucket_to_action` function
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
            return [check_bucket] + list(range(min_bet_bucket, max_bet_bucket+1)) + [14]

        # what if it called?
        elif last_action_taken_by_opponent.type == 'call':
            raise ValueError('This case shouldn\'t happen because a call should lead to the next betting round')

        # what if it bet ?
        elif last_action_taken_by_opponent.type == 'bet':
            call_bucket = get_call_bucket(last_action_taken_by_opponent.value)
            assert opponent_side_pot == last_action_taken_by_opponent.value
            min_raise_bucket = get_min_raise_bucket(last_action_taken_by_opponent.value, actions, b_round, player)
            max_raise_bucket = get_max_bet_bucket(player.stack)
            assert min_raise_bucket > call_bucket, 'buckets are not well calibrated: a bet and a raise can be in the same bucket'
            if max_raise_bucket < call_bucket:
                # in this case, all your money is below the bet of the opponent, so you can only fold or go all-in
                return [-1, 14]
            else:
                return [-1] + [call_bucket] + list(range(min_raise_bucket, max_raise_bucket+1)) + [14]

        # what if it raised ?
        elif last_action_taken_by_opponent.type == 'raise':
            # you have right to do at most 2 min-raises (simplification), and then you have to double at least
            call_bucket = get_call_bucket(last_action_taken_by_opponent.value + opponent_side_pot)
            min_raise_bucket = get_min_raise_bucket(opponent_side_pot, actions, b_round, player, raise_val=last_action_taken_by_opponent.value)
            max_raise_bucket = get_max_bet_bucket(player.stack)
            assert min_raise_bucket > call_bucket, 'buckets are not well calibrated: a bet and a raise can be in the same bucket'
            if max_raise_bucket < call_bucket:
                return [-1, 14]
            else:
                return [-1] + [call_bucket] + list(range(min_raise_bucket, max_raise_bucket+1)) + [14]

        # what if it is all-in ?
        elif last_action_taken_by_opponent.type == 'all in':
            call_bucket = get_call_bucket(opponent_side_pot)
            max_bet_bucket = get_max_bet_bucket(player.stack)
            if max_bet_bucket < call_bucket:
                return [-1, 14]
            else:
                return [-1, call_bucket, 14]

    # in this case, you're first to play and can do basically whatever you want except folding and betting less than the big blind
    except IndexError:
        max_bet_bucket = get_max_bet_bucket(player.stack)
        return [0] + list(range(2, max_bet_bucket+1)) + [14]
