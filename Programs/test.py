from game_utils import *
from actions import *
from strategies import strategy_limper
from nose.tools import *


def get_actions():
    return {b_round: {player: [] for player in range(2)} for b_round in range(4)}


def get_players():
    return [Player(0, strategy_limper, 100, verbose=True, name='A'),
            Player(1, strategy_limper, 100, verbose=True, name='B')]


def test_blinds():
    # A BB with less than the BB, B is SB
    players = get_players()
    players[1].is_dealer = True
    pot = blinds(players)
    assert players[0].side_pot == 1
    assert players[1].side_pot == 1
    assert pot == 2

    # A SB with only one SB, B BB
    players = get_players()
    players[0].is_dealer = True
    pot = blinds(players)
    assert players[0].side_pot == 1
    assert players[1].side_pot == 2
    assert pot == 3

    # Normal conditions
    players = get_players()
    players[0].is_dealer = True
    pot = blinds(players)
    assert players[0].side_pot == 1
    assert players[1].side_pot == 2
    assert pot == 3


def test_bucket_to_action():
    # preflop, you're sb, you fold
    players = get_players()
    bucket = -1
    actions = get_actions()
    player = players[0]
    player.is_dealer = True
    b_round = 0
    blinds(players)
    opponent_side_pot = players[1].side_pot
    action = bucket_to_action(bucket, actions, b_round, player, opponent_side_pot)
    assert action == Action('fold')

    # Preflop, you are SB, you raise the BB
    players = get_players()
    bucket = 10
    actions = get_actions()
    player = players[0]
    player.is_dealer = True
    b_round = 0
    blinds(players)
    opponent_side_pot = players[1].side_pot
    action = bucket_to_action(bucket, actions, b_round, player, opponent_side_pot)
    print(action)
    assert action == Action('raise', 29, min_raise=False), action

    # preflop, you're BB, you call the raise of the SB
    players = get_players()
    bucket = 3
    actions = get_actions()
    player = players[0]
    actions[0][1].append(Action('raise', 2, min_raise=True))
    players[1].is_dealer = True
    b_round = 0
    blinds(players)
    players[1].stack -= 3
    players[1].side_pot += 3
    opponent_side_pot = players[1].side_pot
    action = bucket_to_action(bucket, actions, b_round, player, opponent_side_pot)
    print(action)
    assert action == Action('call', 2), action

    # preflop, you're BB, you go all-in after the raise of the SB
    players = get_players()
    bucket = 14
    actions = get_actions()
    player = players[0]
    actions[0][1].append(Action('raise', 2, min_raise=True))
    players[1].is_dealer = True
    b_round = 0
    blinds(players)
    players[1].stack -= 3
    players[1].side_pot += 3
    opponent_side_pot = players[1].side_pot
    action = bucket_to_action(bucket, actions, b_round, player, opponent_side_pot)
    print(action)
    assert action == Action('all in', player.stack), action

    # flop, you're BB, you bet 10, the SB raised you, and you go all-in
    players = get_players()
    bucket = 14
    actions = get_actions()
    player = players[0]
    actions[1][0].append(Action('bet', 10))
    actions[1][1].append(Action('raise', 10, min_raise=True))
    players[1].is_dealer = True
    b_round = 1
    players[1].stack -= 20
    players[1].side_pot += 20
    players[0].stack -= 10
    players[0].side_pot += 10
    opponent_side_pot = players[1].side_pot
    action = bucket_to_action(bucket, actions, b_round, player, opponent_side_pot)
    print(action)
    assert action == Action('all in', player.stack), action

    # flop, you are SB, you call the bet of the BB
    players = get_players()
    bucket = 5
    actions = get_actions()
    player = players[0]
    actions[1][1].append(Action('bet', 9, min_raise=True))
    players[0].is_dealer = True
    b_round = 1
    players[1].stack -= 9
    players[1].side_pot += 9
    opponent_side_pot = players[1].side_pot
    action = bucket_to_action(bucket, actions, b_round, player, opponent_side_pot)
    assert action == Action('call', 9), action


def test_get_raise_from_bucket():
    # in this case, your SB, nobody played yet, and you can raise at least 2
    players = get_players()
    bucket = 5
    actions = get_actions()
    b_round = 0
    player = players[0]
    player.is_dealer = True
    blinds(players)
    opponent_side_pot = players[1].side_pot
    raise_val = get_raise_from_bucket(bucket, actions, b_round, player, opponent_side_pot, raise_val=0)
    assert raise_val == 5, raise_val

    # in this case, your SB, nobody played yet, and you can raise at least 2
    players = get_players()
    bucket = 3
    actions = get_actions()
    b_round = 0
    player = players[0]
    player.is_dealer = True
    blinds(players)
    opponent_side_pot = players[1].side_pot
    raise_val = get_raise_from_bucket(bucket, actions, b_round, player, opponent_side_pot, raise_val=0)
    assert raise_val == 2, raise_val

    # in this case, flop, raised, you double
    players = get_players()
    bucket = 5
    actions = get_actions()
    actions[1][0].append(Action('bet', 2))
    actions[1][1].append(Action('raise', 2, min_raise=True))
    b_round = 1
    player = players[0]
    players[1].is_dealer = True
    blinds(players)
    players[0].side_pot = 2
    players[1].side_pot = 4
    opponent_side_pot = players[1].side_pot
    raise_val = get_raise_from_bucket(bucket, actions, b_round, player, opponent_side_pot, raise_val=2)
    assert raise_val == 3, raise_val

    # in this case, flop, raised, you triple
    players = get_players()
    bucket = 6
    actions = get_actions()
    actions[1][0].append(Action('bet', 2))
    actions[1][1].append(Action('raise', 2, min_raise=True))
    b_round = 1
    player = players[0]
    players[1].is_dealer = True
    blinds(players)
    players[0].side_pot = 2
    players[1].side_pot = 4
    opponent_side_pot = players[1].side_pot
    raise_val = get_raise_from_bucket(bucket, actions, b_round, player, opponent_side_pot, raise_val=2)
    assert raise_val == 7, raise_val


def test_get_call_bucket():
    bet = 1
    assert get_call_bucket(bet) == 1
    bet = 2
    assert get_call_bucket(bet) == 2
    bet = 5
    assert get_call_bucket(bet) == 4
    bet = 6
    assert get_call_bucket(bet) == 4
    bet = 8
    assert get_call_bucket(bet) == 5
    bet = 50
    assert get_call_bucket(bet) == 11
    bet = 40
    assert get_call_bucket(bet) == 10
    bet = 92
    assert get_call_bucket(bet) == 13, get_call_bucket(bet)


def test_get_max_bet_bucket():
    stack = 100
    max_bucket = get_max_bet_bucket(stack)
    assert max_bucket == 13

    stack = 150
    max_bucket = get_max_bet_bucket(stack)
    assert max_bucket == 13

    stack = 10
    max_bucket = get_max_bet_bucket(stack)
    assert max_bucket == 5

    stack = 1
    max_bucket = get_max_bet_bucket(stack)
    assert max_bucket == 1

    stack = 23
    max_bucket = get_max_bet_bucket(stack)
    assert max_bucket == 8

    stack = 14
    max_bucket = get_max_bet_bucket(stack)
    assert max_bucket == 6


def test_get_min_raise_bucket():
    # if you are small blind and you have to play your first move preflop, you cannot raise to 3, but to 4 at least
    actions = get_actions()
    players = get_players()
    player = players[0]
    player.is_dealer = True
    blinds(players)
    b_round = 0
    bucket = get_min_raise_bucket(players[1].side_pot, actions, b_round, player, raise_val=0)
    assert bucket == 3, bucket

    # if you minraised twice, you don't have right to minraise again
    actions[0][0].append(Action('raise', 2, min_raise=True))
    actions[0][1].append(Action('raise', 2, min_raise=True))
    actions[0][0].append(Action('raise', 2, min_raise=True))
    actions[0][1].append(Action('raise', 2, min_raise=True))
    players[1].side_pot = 10
    players[0].side_pot = 8
    # the pot is therefore 18 at this point (8 for hero vs 10 for villain). You minraised twice
    bucket = get_min_raise_bucket(players[1].side_pot, actions, b_round, player, raise_val=2)
    assert bucket == 8, bucket

    # if you minraised less than twice, you can minraise
    actions = get_actions()
    actions[0][0].append(Action('raise', 2, min_raise=True))
    actions[0][1].append(Action('raise', 2, min_raise=True))
    players[1].side_pot = 6
    players[0].side_pot = 4
    # here the pot is 4 (hero) vs 6 (villain)
    bucket = get_min_raise_bucket(players[1].side_pot, actions, b_round, player, raise_val=2)
    assert bucket == 5, bucket

    # if you minraised less than twice, you can minraise
    actions = get_actions()
    actions[0][0].append(Action('raise', 2, min_raise=True))  # 2-4 (villain-hero side pots)
    actions[0][1].append(Action('raise', 2, min_raise=True))  # 6-4
    actions[0][0].append(Action('raise', 6, min_raise=False))  # 6-12
    actions[0][1].append(Action('raise', 6, min_raise=True))  # 18-12
    players[1].side_pot = 18
    players[0].side_pot = 12
    # here the pot is 12 (hero) vs 18 (villain). You can raise at least to 24
    bucket = get_min_raise_bucket(players[1].side_pot, actions, b_round, player, raise_val=6)
    assert bucket == 8, bucket

    # if you minraised more than twice, you can minraise
    actions = get_actions()
    actions[0][0].append(Action('raise', 2, min_raise=True))  # 2-4
    actions[0][1].append(Action('raise', 2, min_raise=True))  # 6-4
    actions[0][0].append(Action('raise', 6, min_raise=False))  # 6-12
    actions[0][1].append(Action('raise', 6, min_raise=True))  # 18-12
    actions[0][0].append(Action('raise', 6, min_raise=True))  # 18-24
    actions[0][1].append(Action('raise', 6, min_raise=True))  # 30-24
    players[1].side_pot = 30
    players[0].side_pot = 24
    # here the pot is 24 (hero) vs 30 (villain). You can raise at least to 60
    bucket = get_min_raise_bucket(players[1].side_pot, actions, b_round, player, raise_val=6)
    assert bucket == 11, bucket

    # a possible minraise and a raise in the same bucket
    actions = get_actions()
    players = get_players()
    players[1].is_dealer = True
    blinds(players)
    actions[0][1].append(Action('raise', 3, min_raise=False))
    players[1].side_pot = 5
    players[0].side_pot = 2
    # here the pot is 2 (hero) vs 5 (villain). You can raise at least to 8
    bucket = get_min_raise_bucket(players[1].side_pot, actions, b_round, player, raise_val=3)
    assert bucket == 5, bucket


def test_authorized_actions_buckets():
    # preflop, SB
    actions = get_actions()
    players = get_players()
    player = players[0]
    player.is_dealer = True
    blinds(players)
    b_round = 0
    opponent_side_pot = players[1].side_pot
    authorized_actions = authorized_actions_buckets(player, actions, b_round, opponent_side_pot)
    assert authorized_actions == [-1] + list(range(2, 15)), authorized_actions

    # preflop, BB called
    actions = get_actions()
    actions[0][1].append(Action('call', 1))
    players = get_players()
    player = players[0]
    players[1].is_dealer = True
    blinds(players)
    players[1].side_pot += 1
    b_round = 0
    opponent_side_pot = players[1].side_pot
    authorized_actions = authorized_actions_buckets(player, actions, b_round, opponent_side_pot)
    assert authorized_actions == [0] + list(range(3, 15))

    # preflop, BB raised
    actions = get_actions()
    actions[0][1].append(Action('raise', 4, min_raise=False))
    players = get_players()
    player = players[0]
    players[1].is_dealer = True
    blinds(players)
    players[1].side_pot += 4 + 1
    b_round = 0
    opponent_side_pot = players[1].side_pot
    authorized_actions = authorized_actions_buckets(player, actions, b_round, opponent_side_pot)
    assert authorized_actions == [-1] + list(range(4, 15))

    # preflop, BB raised, you raised, it raised again
    actions = get_actions()
    actions[0][1].append(Action('raise', 4, min_raise=False))
    actions[0][0].append(Action('raise', 4, min_raise=True))
    actions[0][1].append(Action('raise', 8, min_raise=False))
    players = get_players()
    player = players[0]
    players[1].is_dealer = True
    blinds(players)
    players[1].side_pot += 1 + 4 + 4 + 8  # 18 in total
    players[0].side_pot += 4 + 4  # 10 in total
    b_round = 0
    opponent_side_pot = players[1].side_pot
    authorized_actions = authorized_actions_buckets(player, actions, b_round, opponent_side_pot)
    assert authorized_actions == [-1, 7] + list(range(9, 15)), authorized_actions  # you can raise at least 8 so the min raise bucket is 9 and not 8

    # flop, SB first play
    actions = get_actions()
    players = get_players()
    player = players[0]
    players[1].is_dealer = True
    b_round = 1
    opponent_side_pot = players[1].side_pot
    authorized_actions = authorized_actions_buckets(player, actions, b_round, opponent_side_pot)
    assert authorized_actions == [0] + list(range(2, 15))

    # flop, SB checked, BB all in
    actions = get_actions()
    players = get_players()
    players[0].stack = 110
    players[1].stack = 90
    actions[1][0].append(Action('check'))
    actions[1][1].append(Action('all in', players[1].stack))
    player = players[0]
    players[0].is_dealer = True
    players[1].side_pot += players[1].stack
    players[1].stack = 0
    b_round = 1
    opponent_side_pot = players[1].side_pot
    authorized_actions = authorized_actions_buckets(player, actions, b_round, opponent_side_pot)
    assert authorized_actions == [-1] + list(range(13, 15)), authorized_actions

    # turn, you are BB and the opponent puts you all-in
    actions = get_actions()
    players = get_players()
    players[0].stack = 50
    players[1].stack = 150
    actions[2][1].append(Action('bet', players[0].stack))
    player = players[0]
    players[0].is_dealer = True
    players[1].side_pot += players[0].stack
    players[1].stack -= players[0].stack
    b_round = 2
    opponent_side_pot = players[1].side_pot
    authorized_actions = authorized_actions_buckets(player, actions, b_round, opponent_side_pot)
    assert authorized_actions == [-1, 14], authorized_actions

    # turn, you are BB and the opponent bet less than your stack
    actions = get_actions()
    players = get_players()
    players[0].stack = 50
    players[1].stack = 150
    actions[2][1].append(Action('bet', 20))
    player = players[0]
    players[0].is_dealer = True
    players[1].side_pot += 20
    players[1].stack -= 20
    b_round = 2
    opponent_side_pot = players[1].side_pot
    authorized_actions = authorized_actions_buckets(player, actions, b_round, opponent_side_pot)
    assert authorized_actions == [-1, 8, 10, 11, 14], authorized_actions
