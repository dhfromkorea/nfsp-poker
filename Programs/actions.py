from utils import sample_categorical


class Action:
    """The possible types of actions"""

    BET_BUCKETS = {
        -1: (None, None),  # this is fold
        0: (0, 0),  # this is check
        # 1: (1, 1),
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
            if get_call_bucket(opponent_side_pot) == bucket:
                # this is a call because the bucket contains the value of the side pot of the opponent
                value_to_bet = opponent_side_pot - player.side_pot
                return Action('call', value=value_to_bet)
            elif get_call_bucket(opponent_side_pot) < bucket:
                # this is a raise
                # it can be a min-raise
                raise_value = get_raise_from_bucket(bucket, actions, b_round, player, opponent_side_pot)
                try:
                    if raise_value == actions[b_round][1-player.id][-1].value:
                        return Action('raise', value=raise_value, min_raise=True)
                    else:
                        return Action('raise', value=raise_value, min_raise=False)
                except IndexError:  # in this case, you are small blind and raise the BB
                    assert b_round == 0
                    assert player.is_dealer
                    assert len(actions[0][1-player.id]) == 0
                    if raise_value == 2:
                        return Action('raise', value=raise_value, min_raise=True)
                    else:
                        return Action('raise', value=raise_value, min_raise=False)



def get_raise_from_bucket(bucket, actions, b_round, player, opponent_side_pot, raise_val=0):
    """
    Note that the raise is what you BET ABOVE THE BET OF THE OPPONENT
    :param bucket:
    :param actions:
    :param b_round:
    :param player:
    :param opponent_side_pot:
    :return:
    """
    min_range_of_your_bucket, max_range_of_your_bucket = Action.BET_BUCKETS[bucket]

    # note that `min_raise` refers to the minimum amount of money you have in your side pot if you min-raise your opponent
    min_raise_bucket, min_raise = get_min_raise_bucket(opponent_side_pot, actions, b_round, player, raise_val=raise_val, return_min_raise=True)
    if max_range_of_your_bucket < min_raise:
        raise ValueError('This is not a raise')
    if min_range_of_your_bucket > min_raise:
        try:
            assert min_range_of_your_bucket - opponent_side_pot > actions[b_round][1-player.id][-1].value, (actions, min_raise, min_range_of_your_bucket, opponent_side_pot, actions[b_round][1-player.id][-1].value)
        except IndexError:
            assert min_range_of_your_bucket - opponent_side_pot > 1
        return min_range_of_your_bucket - opponent_side_pot
    else:
        try:
            assert min_raise > actions[b_round][1 - player.id][-1].value
        except IndexError:
            assert min_raise > 1
        return min_raise - opponent_side_pot


def sample_action(idx, probabilities):
    """
    Sample from categorical distribution
    """
    try:
        return idx[sample_categorical(probabilities)]
    except ValueError:
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
    assert 0 < stack <= 200
    if stack == 1:  # you can just go all-in
        return 14
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
        min_raise = 2*opponent_side_pot
        if not return_min_raise:
            return get_call_bucket(min_raise)
        else:
            return get_call_bucket(min_raise), min_raise
    else:
        # you have right to min-raise
        # if you are small blind, your first raise is at least 4
        if player.is_dealer and b_round == 0 and len(actions[0][1-player.id]) == 0:
            if not return_min_raise:
                return 3
            else:
                return 3, 4  # bucket 3, 4 in total

        min_raise = raise_val + opponent_side_pot
        if not return_min_raise:
            return get_call_bucket(min_raise)
        else:
            return get_call_bucket(min_raise), min_raise


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
            if b_round > 0:
                raise ValueError('This case shouldn\'t happen because a call should lead to the next betting round')

            # for preflop you can raise the call of the small blind
            else:
                assert not player.is_dealer, (actions, player.id, player.is_dealer)
                assert len(actions[player.id][0]) == 0, (actions, player.id, player.is_dealer)
                check_bucket = 0
                return [check_bucket] + list(range(3, 15))

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
        if opponent_side_pot == player.side_pot == 0:
            return [0] + list(range(2, max_bet_bucket+1)) + [14]
        else:
            return [-1] + list(range(2, max_bet_bucket+1)) + [14]
