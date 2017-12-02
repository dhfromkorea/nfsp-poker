from game.game_utils import blinds, bucket_to_action, Card, authorized_actions_buckets, get_min_raise_bucket, get_max_bet_bucket, get_call_bucket, get_raise_from_bucket, Action
from players.strategies import strategy_RL, strategy_random
from players.player import Player, NeuralFictitiousPlayer
from models.q_network import QNetwork, PiNetwork, CardFeaturizer1
from nose.tools import *
from odds.evaluation import evaluate_hand


def get_actions():
    return {b_round: {player: [] for player in range(2)} for b_round in range(4)}


def get_players():
    NUM_HIDDEN_LAYERS = 10
    NUM_ACTIONS = 16
    f = CardFeaturizer1(NUM_HIDDEN_LAYERS, 20)
    Q0 = QNetwork(NUM_ACTIONS, NUM_HIDDEN_LAYERS, f, None, 0, None)
    Q1 = QNetwork(NUM_ACTIONS, NUM_HIDDEN_LAYERS, f, None, 1, None)
    players = [Player(0, strategy_RL(Q0, True), 100, verbose=True, name='SB'),
               Player(1, strategy_RL(Q1, True), 100, verbose=True, name='DH')]
    return players


def c(c_str):
    if len(c_str) == 2:
        return Card(c_str[0], c_str[1])
    else:
        return Card('10', c_str[-1])


def test_blinds():
    # A BB with less than the BB, B is SB
    players = get_players()
    players[0].stack = 1
    players[1].stack = 199
    players[1].is_dealer = True
    pot = blinds(players)
    assert players[0].side_pot == 1, players[0].side_pot
    assert players[1].side_pot == 1, players[1].side_pot
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
    players[0].is_dealer = True
    b_round = 0
    blinds(players)
    opponent_side_pot = players[1].side_pot
    action = bucket_to_action(bucket, actions, b_round, players[0], opponent_side_pot)
    assert action == Action('raise', 30, min_raise=False), action

    # preflop, you're BB, you call the raise of the SB
    players = get_players()
    bucket = 3
    actions = get_actions()
    actions[0][1].append(Action('raise', 2, min_raise=True))  # 4 v 2
    players[1].is_dealer = True
    b_round = 0
    blinds(players)
    players[1].stack -= 3
    players[1].side_pot += 3
    opponent_side_pot = players[1].side_pot
    action = bucket_to_action(bucket, actions, b_round, players[0], opponent_side_pot)
    assert action == Action('raise', 2, min_raise=True), action

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
    assert action == Action('all in', player.stack), action

    # flop, you are SB, you call the bet of the BB
    players = get_players()
    bucket = 5
    actions = get_actions()
    player = players[0]
    actions[1][1].append(Action('bet', 9))
    players[0].is_dealer = True
    b_round = 1
    players[1].stack -= 9
    players[1].side_pot += 9
    opponent_side_pot = players[1].side_pot
    action = bucket_to_action(bucket, actions, b_round, player, opponent_side_pot)
    assert action == Action('call', 9), action

    # preflop, you are SB, you raise 40, the opponent raises 40, you call
    players = get_players()
    bucket = 10  # call 40
    actions = get_actions()
    player = players[0]
    actions[0][0].append(Action('raise', 40, min_raise=False))
    actions[0][1].append(Action('raise', 40, min_raise=True))
    players[0].is_dealer = True
    b_round = 0
    players[0].stack = 101 - 1 - 40
    players[1].stack = 99 - 2 - 40
    players[0].side_pot = 42
    players[1].side_pot = 82
    opponent_side_pot = players[1].side_pot
    action = bucket_to_action(bucket, actions, b_round, player, opponent_side_pot)
    assert action == Action('call', 40), action


def test_get_raise_from_bucket():
    # in this case, your SB, nobody played yet, and you can raise at least 2
    players = get_players()
    bucket = 5
    actions = get_actions()
    b_round = 0
    players[0].is_dealer = True
    blinds(players)
    opponent_side_pot = players[1].side_pot
    raise_val = get_raise_from_bucket(bucket, actions, b_round, players[0], opponent_side_pot, raise_val=0)
    assert raise_val == 6, raise_val

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
    players[1].is_dealer = True
    blinds(players)
    players[0].side_pot = 2
    players[1].side_pot = 4
    opponent_side_pot = players[1].side_pot
    raise_val = get_raise_from_bucket(bucket, actions, b_round, players[0], opponent_side_pot, raise_val=2)
    assert raise_val == 5, raise_val

    # in this case, flop, raised, you triple
    players = get_players()
    bucket = 6
    actions = get_actions()
    actions[1][0].append(Action('bet', 2))
    actions[1][1].append(Action('raise', 2, min_raise=True))
    b_round = 1
    players[1].is_dealer = True
    blinds(players)
    players[0].side_pot = 2
    players[1].side_pot = 4
    opponent_side_pot = players[1].side_pot
    raise_val = get_raise_from_bucket(bucket, actions, b_round, players[0], opponent_side_pot, raise_val=2)
    assert raise_val == 9, raise_val

    #
    actions = get_actions()
    players = get_players()
    players[0].is_dealer = True
    blinds(players)
    actions[0][0].append(Action('raise', 9, min_raise=False))  # 11 v 2
    actions[0][1].append(Action('raise', 9, min_raise=True))  # 11 v 20
    players[0].side_pot = 11
    players[0].stack = 89
    players[1].side_pot = 20
    players[1].stack = 80
    bucket = 8
    # it chooses bucket 8, i.e 20, i.e raise of 11
    raise_val = get_raise_from_bucket(bucket, actions, 0, players[0], players[1].side_pot, raise_val=9)
    assert raise_val == 11, raise_val


def test_get_call_bucket():
    bet = 2
    assert get_call_bucket(bet) == 2, get_call_bucket(bet)
    bet = 5
    assert get_call_bucket(bet) == 4, get_call_bucket(bet)
    bet = 6
    assert get_call_bucket(bet) == 4, get_call_bucket(bet)
    bet = 8
    assert get_call_bucket(bet) == 5, get_call_bucket(bet)
    bet = 50
    assert get_call_bucket(bet) == 11, get_call_bucket(bet)
    bet = 40
    assert get_call_bucket(bet) == 10, get_call_bucket(bet)
    bet = 92
    assert get_call_bucket(bet) == 13, get_call_bucket(bet)


def test_get_max_bet_bucket():
    stack = 100
    max_bucket = get_max_bet_bucket(stack)
    assert max_bucket == 13, max_bucket

    stack = 150
    max_bucket = get_max_bet_bucket(stack)
    assert max_bucket == 13, max_bucket

    stack = 10
    max_bucket = get_max_bet_bucket(stack)
    assert max_bucket == 5, max_bucket

    stack = 1
    max_bucket = get_max_bet_bucket(stack)
    assert max_bucket == 1, max_bucket

    stack = 23
    max_bucket = get_max_bet_bucket(stack)
    assert max_bucket == 8, max_bucket

    stack = 14
    max_bucket = get_max_bet_bucket(stack)
    assert max_bucket == 6, max_bucket


def test_get_min_raise_bucket():
    # if you are small blind and you have to play your first move preflop, you cannot raise to 3, but to 4 at least
    actions = get_actions()
    players = get_players()
    players[0].is_dealer = True
    blinds(players)
    b_round = 0
    bucket = get_min_raise_bucket(players[1].side_pot, actions, b_round, players[0], raise_val=0)
    assert bucket == 3, bucket

    # if you minraised twice, you don't have right to minraise again
    actions[0][0].append(Action('raise', 2, min_raise=True))
    actions[0][1].append(Action('raise', 2, min_raise=True))
    actions[0][0].append(Action('raise', 2, min_raise=True))
    actions[0][1].append(Action('raise', 2, min_raise=True))
    players[1].side_pot = 10
    players[0].side_pot = 8
    # the pot is therefore 18 at this point (8 for hero vs 10 for villain). You minraised twice
    # now you can raise at min 10 (i.e to 20), i.e bet 12, i.e bucket 6
    bucket = get_min_raise_bucket(players[1].side_pot, actions, b_round, players[0], raise_val=2)
    assert bucket == 6, bucket

    # if you minraised less than twice, you can minraise
    actions = get_actions()
    actions[0][0].append(Action('raise', 2, min_raise=True))
    actions[0][1].append(Action('raise', 2, min_raise=True))
    players[1].side_pot = 6
    players[0].side_pot = 4
    # here the pot is 4 (hero) vs 6 (villain). you can min raise 2, i.8 to 8, i.e bet 4, i.e bucket 3
    bucket = get_min_raise_bucket(players[1].side_pot, actions, b_round, players[0], raise_val=2)
    assert bucket == 3, bucket

    # if you minraised less than twice, you can minraise
    actions = get_actions()
    actions[0][0].append(Action('raise', 2, min_raise=True))  # 2-4 (villain-hero side pots)
    actions[0][1].append(Action('raise', 2, min_raise=True))  # 6-4
    actions[0][0].append(Action('raise', 6, min_raise=False))  # 6-12
    actions[0][1].append(Action('raise', 6, min_raise=True))  # 18-12
    players[1].side_pot = 18
    players[0].side_pot = 12
    # here the pot is 12 (hero) vs 18 (villain). You can raise at least 6, i.e to 24, i.e bet 12, i.e bucket 6
    bucket = get_min_raise_bucket(players[1].side_pot, actions, b_round, players[0], raise_val=6)
    assert bucket == 6, bucket

    # if you minraised twice, you can no longer minraise
    actions = get_actions()
    actions[0][0].append(Action('raise', 2, min_raise=True))  # 2-4
    actions[0][1].append(Action('raise', 2, min_raise=True))  # 6-4
    actions[0][0].append(Action('raise', 6, min_raise=False))  # 6-12
    actions[0][1].append(Action('raise', 6, min_raise=True))  # 18-12
    actions[0][0].append(Action('raise', 6, min_raise=True))  # 18-24
    actions[0][1].append(Action('raise', 6, min_raise=True))  # 30-24
    players[1].side_pot = 30
    players[0].side_pot = 24
    # here the pot is 24 (hero) vs 30 (villain). You can raise at least 30, i.e bet 36, i.e bucket 10
    bucket = get_min_raise_bucket(players[1].side_pot, actions, b_round, players[0], raise_val=6)
    assert bucket == 10, bucket

    # a possible minraise and a raise in the same bucket
    actions = get_actions()
    players = get_players()
    players[1].is_dealer = True
    blinds(players)
    actions[0][1].append(Action('raise', 3, min_raise=False))
    players[1].side_pot += 4
    players[1].stack -= 4
    players[0].side_pot = 2
    # here the pot is 2 (hero) vs 5 (villain). You can raise at least 3, i.e to 8, i.e bet 6, i.e bucket 4
    bucket = get_min_raise_bucket(players[1].side_pot, actions, b_round, players[0], raise_val=3)
    assert bucket == 4, bucket


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
    assert authorized_actions == [-1, 1] + list(range(3, 15)), authorized_actions

    # preflop, you are the BB, the SB called
    actions = get_actions()
    actions[0][1].append(Action('call', 1))
    players = get_players()
    players[1].is_dealer = True
    blinds(players)
    players[1].side_pot += 1
    b_round = 0
    opponent_side_pot = players[1].side_pot
    authorized_actions = authorized_actions_buckets(players[0], actions, b_round, opponent_side_pot)
    assert authorized_actions == [0] + list(range(2, 15)), authorized_actions

    # preflop, BB raised
    actions = get_actions()
    actions[0][1].append(Action('raise', 4, min_raise=False))  # 2 v 6
    players = get_players()
    players[1].is_dealer = True
    blinds(players)
    players[1].side_pot += 4 + 1
    b_round = 0
    opponent_side_pot = players[1].side_pot
    # you can call (i.e bet 4, i.e bucket 3), or raise at least 4 (i.e bet 8, i.e bucket 5)
    authorized_actions = authorized_actions_buckets(players[0], actions, b_round, opponent_side_pot)
    assert authorized_actions == [-1, 3] + list(range(5, 15)), authorized_actions

    # preflop, BB raised, you raised, it raised again
    actions = get_actions()
    actions[0][1].append(Action('raise', 4, min_raise=False))
    actions[0][0].append(Action('raise', 4, min_raise=True))
    actions[0][1].append(Action('raise', 8, min_raise=False))
    players = get_players()
    players[1].is_dealer = True
    blinds(players)
    players[1].side_pot += 1 + 4 + 4 + 8  # 18 in total
    players[0].side_pot += 4 + 4  # 10 in total
    b_round = 0
    opponent_side_pot = players[1].side_pot
    # you can either call (i.e bet 8, i.e bucket 5) or raise at least 8 (i.e bet 16, i.e bucket 7)
    authorized_actions = authorized_actions_buckets(players[0], actions, b_round, opponent_side_pot)
    assert authorized_actions == [-1, 5] + list(range(7, 15)), authorized_actions  # you can raise at least 8 so the min raise bucket is 9 and not 8

    # flop, SB first play
    actions = get_actions()
    players = get_players()
    player = players[0]
    players[1].is_dealer = True
    b_round = 1
    opponent_side_pot = players[1].side_pot
    authorized_actions = authorized_actions_buckets(player, actions, b_round, opponent_side_pot)
    assert authorized_actions == [0] + list(range(2, 15)), authorized_actions

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
    assert authorized_actions == [-1, 13], authorized_actions

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

    # flop, you have 129 v 63, there is 8 in the pot. The opponent is dealer
    # you bet 61, it goes all-in 63
    actions = get_actions()
    players = get_players()
    players[0].stack = 129
    players[1].stack = 63
    actions[1][0].append(Action('bet', 61))
    actions[1][1].append(Action('all in', 63))
    players[1].is_dealer = True
    players[1].stack = 0
    players[0].stack = 68
    players[1].side_pot = 63
    players[0].side_pot = 61
    b_round = 1
    opponent_side_pot = players[1].side_pot
    authorized_actions = authorized_actions_buckets(players[0], actions, b_round, opponent_side_pot)
    # you can either fold or call
    assert authorized_actions == [-1, 2], authorized_actions

    # flop. The pot is 102. You have 43, your opponent 35. The pot is 122
    # You bet 20. The opponent can either fold, call or go all-in
    actions = get_actions()
    players = get_players()
    players[0].stack = 23
    players[1].stack = 35
    actions[1][0].append(Action('bet', 20))
    players[1].is_dealer = True
    players[1].side_pot = 0
    players[0].side_pot = 20
    b_round = 1
    opponent_side_pot = players[0].side_pot
    authorized_actions = authorized_actions_buckets(players[1], actions, b_round, opponent_side_pot)
    # you can either fold or call
    assert authorized_actions == [-1, 8, 14], authorized_actions

    # Preflop. You have 22, your opponent 178. You are dealer
    # you raise 2 (to 4), it raise 4 (to 8). You have 13 left
    actions = get_actions()
    players = get_players()
    players[0].stack = 13
    players[1].stack = 168
    actions[0][0].append(Action('raise', 2, min_raise=True))
    actions[0][1].append(Action('raise', 4, min_raise=False))
    players[0].is_dealer = True
    players[1].side_pot = 8
    players[0].side_pot = 4
    b_round = 0
    opponent_side_pot = players[1].side_pot
    authorized_actions = authorized_actions_buckets(players[0], actions, b_round, opponent_side_pot)
    # you can either fold, call (i.e bet 4, i.e bucket 3), minraise (to 12, meaning you bet 8 more, i.e bucket 5). You can also bucket 6 (bet 11 or 12). Betting 13 means going all-in
    assert authorized_actions == [-1, 3, 5, 6, 14], authorized_actions

    # Preflop. You have 143, your opponent 53. You are dealer and call
    actions = get_actions()
    players = get_players()
    players[0].stack = 143
    players[1].stack = 53
    actions[0][0].append(Action('call', 1))
    players[0].is_dealer = True
    players[1].side_pot = 2
    players[0].side_pot = 2
    b_round = 0
    opponent_side_pot = players[0].side_pot
    authorized_actions = authorized_actions_buckets(players[1], actions, b_round, opponent_side_pot)
    # you can either check, or minraise (i.e bet 2, i.e bucket 2), raise (to at most 52, i.e bucket 11) You can also go all in
    assert authorized_actions == [0] + list(range(2, 12)) + [14], authorized_actions

    # Preflop. You have 2, your opponent 198. You are dealer/SB
    actions = get_actions()
    players = get_players()
    players[0].stack = 2
    players[1].stack = 198
    players[0].is_dealer = True
    blinds(players, False)
    b_round = 0
    opponent_side_pot = players[1].side_pot
    authorized_actions = authorized_actions_buckets(players[0], actions, b_round, opponent_side_pot)
    # you can either check, or minraise (i.e bet 2, i.e bucket 2), raise (to at most 52, i.e bucket 11) You can also go all in
    assert authorized_actions == [-1, 14], authorized_actions

    # turn, you have 20 left
    actions = get_actions()
    players = get_players()
    players[0].stack = 20
    players[1].stack = 200 - 20 - 26 - 15 - 46
    pot = 2*(26 + 15 + 46)
    players[1].is_dealer = True
    b_round = 2
    opponent_side_pot = 0
    actions[0][0].append(Action('call', 24))
    actions[0][1].append(Action('raise', 25))
    actions[1][0].append(Action('bet', 15))
    actions[1][0].append(Action('call', 46))
    actions[1][1].append(Action('raise', 46))
    authorized_actions = authorized_actions_buckets(players[0], actions, b_round, opponent_side_pot)
    assert authorized_actions == [0] + list(range(2, 8)) + [14], authorized_actions

    # # turn,
    # actions = get_actions()
    # players = get_players()
    # players[0].stack = 101
    # players[1].stack = 99
    # blinds(players)
    # players[0].is_dealer = True
    # b_round = 2
    # actions[0][0].append(Action('raise', 40, min_raise=False))  # 42 v 2
    # actions[0][1].append(Action('raise', 40, min_raise=True))  # 42 v 82
    # actions[0][0].append(Action('call', 38))  # PROBLEM IT IS NOT SUPPOSED TO HAPPEN
    # actions[1][1].append(Action('bet', 11))
    # actions[1][0].append(Action('call', 11))
    # actions[2][1].append(Action('check', 0))
    #
    # players[0].side_pot = 0
    # players[1].side_pot = 0
    # players[0].stack = 101 - 42 - 38
    # players[1].stack = 0
    #
    # authorized_actions = authorized_actions_buckets(players[0], actions, b_round, opponent_side_pot)
    # assert authorized_actions == [0] + list(range(2, 8)) + [14], authorized_actions

    # river, you have 2 left and go all in
    actions = get_actions()
    players = get_players()
    players[0].is_dealer = True
    blinds(players)
    b_round = 3
    actions[0][0].append(Action('raise', 30, min_raise=False))  # 32 v 2
    actions[0][1].append(Action('call', 30))  # 32 v 32
    actions[1][1].append(Action('bet', 2))  # 0 v 2
    actions[1][0].append(Action('raise', 29, min_raise=False))  # 31 v 2
    actions[1][1].append(Action('raise', 29, min_raise=True))  # 31 v 60
    actions[1][0].append(Action('call', 29))
    actions[2][1].append(Action('bet', 2))  # 0 v 2
    actions[2][0].append(Action('raise', 5))  # 7 v 2
    actions[2][1].append(Action('call', 5))  # 7 v 7
    actions[3][1].append(Action('check', 0))  # 0 v 0

    players[0].side_pot = 0
    players[1].side_pot = 0
    players[0].stack = 2
    opponent_side_pot = 0
    b_round = 3

    authorized_actions = authorized_actions_buckets(players[0], actions, b_round, opponent_side_pot)
    assert authorized_actions == [0, 14], authorized_actions

    # flop, you have 41 left and go all in
    actions = get_actions()
    players = get_players()
    players[1].is_dealer = True
    blinds(players)
    b_round = 1
    actions[1][0].append(Action('check', 0))  # 0v0

    players[0].side_pot = 0
    players[1].side_pot = 0
    players[0].stack = 41
    players[1].stack = 41
    opponent_side_pot = 41
    authorized_actions = authorized_actions_buckets(players[1], actions, b_round, opponent_side_pot)
    assert authorized_actions == [0] + list(range(2,11)) + [14], authorized_actions


def test_all_actions_preflop():
    # FIRST SITUATION
    actions = get_actions()
    players = get_players()
    players[1].is_dealer = True
    blinds(players)

    actions[0][1].append(Action('raise', 3, min_raise=False))  # 5 v 2
    actions[0][0].append(Action('raise', 22, min_raise=False))  # 5 v 27
    # it can either call 22, or raise at least 22 (to 49)
    players[0].side_pot = 27
    players[1].side_pot = 5
    players[0].stack -= 27
    players[1].stack -= 5
    possible_actions = authorized_actions_buckets(players[1], actions, 0, players[0].side_pot)
    assert possible_actions == [-1, 8, 11, 12, 13, 14], possible_actions

    # SECOND SITUATION
    actions = get_actions()
    players = get_players()
    players[1].is_dealer = True
    blinds(players)

    actions[0][1].append(Action('raise', 2, min_raise=True))  # 4 v 2
    # it can either call 4 (i.e bet 2, i.e bucket 2), or raise at least 2 (to 6, i.e bet 4, i.e bucket 3)
    players[1].side_pot += 3
    players[1].stack -= 3
    possible_actions = authorized_actions_buckets(players[0], actions, 0, players[1].side_pot)
    assert possible_actions == [-1] + list(range(2, 15)), possible_actions

    # THIRD SITUATION
    actions = get_actions()
    players = get_players()
    players[0].is_dealer = True
    blinds(players)
    actions[0][0].append(Action('raise', 3, min_raise=False))  # 5 v 2
    actions[0][1].append(Action('raise', 9, min_raise=False))  # 5 v 14
    actions[0][0].append(Action('raise', 44, min_raise=False))  # 58 v 14
    # it can either call 44 (i.e bet 44, i.e bucket 11), or fo all-in
    players[1].side_pot += 1 + 3 + 9
    players[1].stack -= 1 + 3 + 9  # 86
    players[0].side_pot = 58
    players[0].stack = 42
    possible_actions = authorized_actions_buckets(players[1], actions, 0, players[0].side_pot)
    assert possible_actions == [-1] + [11, 14], possible_actions

    # FOURTH SITUATION
    actions = get_actions()
    players = get_players()
    players[0].is_dealer = True
    blinds(players)
    actions[0][0].append(Action('raise', 18, min_raise=False))  # 20 v 2
    # it can either call 18 (i.e bet 18, i.e bucket 7), or raise at least 18 (i.e to 38, i.e bet 36, i.e bucket 10)
    players[0].side_pot = 20
    players[0].stack = 80
    possible_actions = authorized_actions_buckets(players[1], actions, 0, players[0].side_pot)
    assert possible_actions == [-1, 7] + list(range(10, 15)), possible_actions

    # FIFTH SITUATION
    actions = get_actions()
    players = get_players()
    players[0].is_dealer = True
    blinds(players)
    actions[0][0].append(Action('raise', 29, min_raise=False))  # 31 v 2
    actions[0][1].append(Action('raise', 29, min_raise=True))  # 31 v 60
    players[0].side_pot = 31
    players[0].stack = 69
    players[1].side_pot = 60
    players[1].stack = 40
    # it can either call 29 (bucket 9) or raise 29 (i.e to 89, i.e bet 58, i.e bucket 11)
    possible_actions = authorized_actions_buckets(players[0], actions, 0, players[1].side_pot)
    assert possible_actions == [-1, 9] + list(range(11, 13)) + [14], possible_actions

    # SIXTH sITUATION
    actions = get_actions()
    players = get_players()
    players[0].is_dealer = True
    blinds(players)
    actions[0][0].append(Action('raise', 9, min_raise=False))  # 11 v 2
    actions[0][1].append(Action('raise', 9, min_raise=True))  # 11 v 20
    players[0].side_pot = 11
    players[0].stack = 89
    players[1].side_pot = 20
    players[1].stack = 80
    # it can either call 9 (bucket 5) or raise 9 (i.e to 29, i.e bet 18, i.e bucket 7)
    possible_actions = authorized_actions_buckets(players[0], actions, 0, players[1].side_pot)
    assert possible_actions == [-1, 5] + list(range(7, 15)), possible_actions

    # 7th situation
    actions = get_actions()
    players = get_players()
    players[0].is_dealer = True
    players[0].stack = 66
    players[1].stack = 134
    blinds(players)
    actions[0][0].append(Action('raise', 4, min_raise=False))  # 6 v 2
    actions[0][1].append(Action('raise', 4, min_raise=True))  # 6 v 10
    actions[0][0].append(Action('raise', 13, min_raise=False))  # 23 v 10
    actions[0][1].append(Action('raise', 48, min_raise=False))  # 23 v 71
    players[0].side_pot = 23
    players[1].side_pot = 71
    players[0].stack = 66 - 1 - 23
    players[1].stack = 134 - 2 - 71
    # it can either go all in or fold
    possible_actions = authorized_actions_buckets(players[0], actions, 0, players[1].side_pot)
    assert possible_actions == [-1, 14], possible_actions

    # 8th situation
    actions = get_actions()
    players = get_players()
    players[1].is_dealer = True
    players[1].stack = 189+1
    players[0].stack = 8+2
    blinds(players)
    actions[0][1].append(Action('raise', 10, min_raise=False))  # 2 v 12
    players[0].side_pot = 2
    players[1].side_pot = 12
    players[0].stack = 8
    players[1].stack = 190-12
    # it can either go all in or fold
    possible_actions = authorized_actions_buckets(players[0], actions, 0, players[1].side_pot)
    assert possible_actions == [-1, 14], possible_actions

    # 9th situation
    actions = get_actions()
    players = get_players()
    players[0].is_dealer = True
    players[1].stack = 100
    players[0].stack = 100
    blinds(players)
    actions[0][0].append(Action('raise', 10, min_raise=False))  # 12 v 2
    actions[0][1].append(Action('all in', 98))  # 12 v 100

    players[0].side_pot = 12
    players[1].side_pot = 100
    players[0].stack = 88
    players[1].stack = 0
    # it can either go all in or fold
    possible_actions = authorized_actions_buckets(players[0], actions, 0, players[1].side_pot)
    assert possible_actions == [-1, 14], possible_actions

    # 10th situation
    actions = get_actions()
    players = get_players()
    players[1].is_dealer = True
    players[1].stack = 104
    players[0].stack = 96
    blinds(players)
    actions[0][1].append(Action('raise', 4, min_raise=False))  # 2 v 6
    actions[0][0].append(Action('raise', 4, min_raise=True))  # 10 v 6
    actions[0][1].append(Action('all in', 98))  # 10 v 104

    players[0].side_pot = 10
    players[1].side_pot = 104
    players[0].stack = 86
    players[1].stack = 0
    # it can either go all in or fold
    possible_actions = authorized_actions_buckets(players[0], actions, 0, players[1].side_pot)
    assert possible_actions == [-1, 14], possible_actions

    # 11th situation
    actions = get_actions()
    players = get_players()
    players[0].is_dealer = True
    players[1].stack = 198
    players[0].stack = 2
    pot = blinds(players)

    players[0].side_pot = 1
    players[1].side_pot = 2
    players[0].stack = 1
    players[1].stack = 196
    # it can either go all in or fold
    possible_actions = authorized_actions_buckets(players[0], actions, 0, players[1].side_pot)
    assert possible_actions == [-1, 14], possible_actions

    # 12th situation
    actions = get_actions()
    players = get_players()
    players[1].is_dealer = True
    players[1].stack = 52
    players[0].stack = 148
    pot = blinds(players)

    actions[0][1].append(Action('raise', 25, min_raise=False))
    actions[0][0].append(Action('raise', 25, min_raise=True))
    actions[0][1].append(Action('all in', 25))
    players[1].side_pot = 52
    players[0].stack = 96
    players[1].stack = 0
    # it can either go all in or fold
    possible_actions = authorized_actions_buckets(players[0], actions, 0, players[1].side_pot)
    assert possible_actions == [-1] + list(range(8, 14)), possible_actions

