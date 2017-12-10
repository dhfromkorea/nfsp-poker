"""
Contains strategy functions

They should all have the same signature, but don't always use all this information
"""
from game.game_utils import *
from game.state import build_state
from game.utils import softmax, variable
import numpy as np
import random
import torch as t
import random
from timeit import default_timer as timer


TARGET_NETWORK_UPDATE_PERIOD = 300  # every 300 episodes
ANTICIPATORY_PARAMETER = 0.1
EPSILON = 0.01
NUM_HIDDEN_LAYERS = 10
NUM_ACTIONS = 16
GAMMA_VAL = 0.95


def get_random_action(possible_actions, actions, b_round, player, opponent_side_pot):
    random_action_bucket = np.random.choice(possible_actions)

    random_action = bucket_to_action(random_action_bucket, actions, b_round, player, opponent_side_pot)
    return random_action


def strategy_random(player, board, pot, actions, b_round, opponent_stack, opponent_side_pot, greedy=True, blinds=BLINDS, verbose=False):
    """
    Take decision randomly amongst any amount of raise, call , fold, all-in or check.
    """
    possible_actions = authorized_actions_buckets(player, actions, b_round, opponent_side_pot)  # you don't have right to take certain actions, e.g betting more than you have or betting 0 or checking a raise
    return get_random_action(possible_actions, actions, b_round, player, opponent_side_pot)


def strategy_mirror(player, board, pot, actions, b_round, opponent_stack, opponent_side_pot, greedy=True, blinds=BLINDS, verbose=False):
    """
    Take decision to level always. Small blind big blind will work as usual.Else, the agent tries to level the board
    always. If the opponent raises, agent calls.If the agent goes first, it checks always if possible. If the opponent checks, agent checks too.
    If opponent goes all in agent goes all in too.
    """
    # In this case, you play after the opponent
    try:
        opponent_id = 1 - player.id
        last_opponent_action = actions[b_round][opponent_id][-1]

        if last_opponent_action.type == "check":
            check_bucket = 0
            mirror_action = bucket_to_action(check_bucket, actions, b_round, player, opponent_side_pot)
            return mirror_action

        elif last_opponent_action.type == "call":
            check_bucket = 0
            mirror_action = bucket_to_action(check_bucket, actions, b_round, player, opponent_side_pot)
            return mirror_action

        elif last_opponent_action.type == "bet":
            call_bucket = get_call_bucket(last_opponent_action.value)
            mirror_action = bucket_to_action(call_bucket, actions, b_round, player, opponent_side_pot)
            return mirror_action

        elif last_opponent_action.type == "all in":
            return Action('all in', player.stack)

        elif last_opponent_action.type == "raise":
            possible_actions = authorized_actions_buckets(player, actions, b_round, opponent_side_pot)[1:]
            action_bucket = possible_actions[0]
            action = bucket_to_action(action_bucket, actions, b_round, player, opponent_side_pot)
            return action

    # You play first. Always check for non pre-flop.
    except IndexError:
        max_bet_bucket = get_max_bet_bucket(player.stack)
        if b_round == 0:  # preflop, you are SB and it is your first action. You can either fold, call, or raise at least 2 (i.e bet 3, i.e bucket 3)
            if max_bet_bucket <= 2:
                small_blind_bucket = 14
            else:
                small_blind_bucket = 1
            mirror_action = bucket_to_action(small_blind_bucket, actions, b_round, player, opponent_side_pot)
            return mirror_action
        else:
            check_bucket = 0
            mirror_action = bucket_to_action(check_bucket, actions, b_round, player, opponent_side_pot)
            return mirror_action


def strategy_RL_aux(player, board, pot, actions, b_round, opponent_stack, opponent_side_pot, Q,
                    greedy=True, blinds=BLINDS, verbose=False, eps=0., cuda=False, for_play=False):
    """
    Take decision using Q values (in a greedy or random way)
    :param player:
    :param board:
    :param pot:
    :param actions:
    :param b_round:
    :param opponent_stack:
    :param Q: the Keras neural network that takes states as inputs
    :param greedy: True for greedy, False for Q-softmax sampling
    :param blinds:
    :param verbose:
    :return:
    """
    possible_actions = authorized_actions_buckets(player, actions, b_round, opponent_side_pot)  # you don't have right to take certain actions, e.g betting more than you have or betting 0 or checking a raise

    # @hack
    # add some heuristics
    # if check is possible
    if 0 in possible_actions and -1 in possible_actions:
        # one should not fold
        del possible_actions[possible_actions.index(-1)]

    if np.all([i in possible_actions for i in [-1, 0]]):
        import pdb;pdb.set_trace()

    state = build_state(player, board, pot, actions, opponent_stack, blinds[1], as_variable=False)
    state = [variable(s, cuda=cuda) for s in state]
    state.append(for_play)
    Q_values = Q.forward(*state)[0].squeeze()  # it has multiple outputs, the first is the Qvalues
    Q_values = Q_values.data.cpu().numpy()

    # choose action in a greedy way
    if greedy:
        Q_values_for_possible_actions = np.array([Q_value for k, Q_value in enumerate(Q_values) if
                                                  idx_to_bucket(k) in possible_actions])
        ties = np.flatnonzero(Q_values_for_possible_actions == Q_values_for_possible_actions.max())
        best_possible_action_bucket = np.random.choice(ties)
        best_possible_action_bucket = [idx_to_bucket(k) for k, Q_value in enumerate(Q_values) if idx_to_bucket(k) in possible_actions][best_possible_action_bucket]
        action = bucket_to_action(best_possible_action_bucket, actions, b_round, player, opponent_side_pot)
    else:
        idx = [idx_to_bucket(k) for k, Q_value in enumerate(Q_values) if idx_to_bucket(k) in possible_actions]
        Q_values = [Q_value for k, Q_value in enumerate(Q_values) if idx_to_bucket(k) in possible_actions]
        probabilities = softmax(Q_values)
        assert np.abs(np.sum(probabilities) - 1.) < 1e-6, probabilities
        action = bucket_to_action(sample_action(idx, probabilities), actions, b_round, player, opponent_side_pot)

    is_epsilon = (random.random() <= eps)
    if is_epsilon:
        return get_random_action(possible_actions, actions, b_round, player, opponent_side_pot)
    else:
        return action


def strategy_RL(Q, greedy):
    """Function generator"""
    return lambda player, board, pot, actions, b_round, opponent_stack, opponent_side_pot, blinds=BLINDS, verbose=False, eps=0: strategy_RL_aux(player, board, pot, actions, b_round, opponent_stack, opponent_side_pot, Q, greedy=greedy, blinds=blinds,
                                                                                                                                                verbose=verbose, eps=eps)


class StrategyNFSP():
    def __init__(self, Q, pi, eta, eps, is_greedy=True, verbose=False, cuda=False):
        self._Q = Q
        self._pi = pi
        self._target_Q = Q
        self.eps = eps
        self.eta = eta
        self.is_Q_used = False
        self.is_greedy = is_greedy
        self.verbose = verbose
        self.cuda = cuda
        self.is_graph_created = False

    def choose_action(self, player, board, pot, actions, b_round, opponent_stack, opponent_side_pot,
                      blinds, episode_idx, for_play=False):
        # decay epsilon in the same way in the paper (NFSP, 2016)
        # we use number of episodes as n
        # the exact decay schedule was not specified but alluded to slower than sqrt
        # so we use (n)^1/4
        self.eps = self.eps / np.power(np.max([episode_idx, 1]), 1/4)
        if player.is_all_in:
            assert player.stack == 0
            return Action('null'), False

        if self.eta >= np.random.rand():
            # use epsilon-greedy policy
            if self.verbose:
                start = timer()

            action = strategy_RL_aux(player, board, pot,
                                     actions, b_round, opponent_stack,
                                     opponent_side_pot, self._Q,
                                     greedy=self.is_greedy,
                                     blinds=blinds, verbose=self.verbose,
                                     eps=self.eps, cuda=self.cuda, for_play=for_play)

            if self.verbose:
                print('forward pass of Q took', timer() - start)

            self.is_Q_used = True
        else:
            # use average policy
            state = build_state(player, board, pot, actions, opponent_stack, blinds[1],
                                as_variable=False)
            state = [variable(s, cuda=self.cuda) for s in state]
            state.append(for_play)
            if self.verbose:
                start = timer()

            action_probs = self._pi.forward(*state).squeeze()

            if self.verbose:
                print('forward pass of pi took', timer() - start)
            possible_actions = authorized_actions_buckets(player, actions, b_round, opponent_side_pot)
            # @hack
            # add some heuristics
            # if check is possible
            if 0 in possible_actions and -1 in possible_actions:
                # one should not fold
                del possible_actions[possible_actions.index(-1)]

            if np.all([i in possible_actions for i in [-1, 0]]):
                import pdb;pdb.set_trace()

            idx = [idx_to_bucket(k) for k, _ in enumerate(action_probs) if idx_to_bucket(k) in possible_actions]
            valid_action_probs = t.stack([p for k, p in enumerate(action_probs) if idx_to_bucket(k) in possible_actions])

            valid_action_probs /= t.sum(valid_action_probs)

            action = bucket_to_action(sample_action(idx, valid_action_probs.data.cpu().numpy()), actions, b_round, player, opponent_side_pot)
            self.is_Q_used = False
        return action, self.is_Q_used

    def sync_target_network(self):
        """
        create a fixed target network
        copy weights and memorys
        """
        try:
            self._target_Q.load_state_dict(self._Q.state_dict())
        except:
            raise Exception('failed to synchronize traget network')
