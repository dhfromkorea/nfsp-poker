from experience_replay.experience_replay import ReplayBufferManager
from game.game_utils import Action
from game.state import build_state, create_state_variable_batch
from game.action import create_action_variable_batch
from game.reward import create_reward_variable_batch
from game.config import BLINDS

import numpy as np

# define some utility functions
create_state_var = create_state_variable_batch()
create_action_var = create_action_variable_batch()
create_reward_var = create_reward_variable_batch()

# define RL hyperparameters here
# chosen to match the NSFP paper

class Player:
    '''
    default player bot
    '''
    def __init__(self, pid, strategy, stack, name=None, verbose=False):
        self.id = pid
        self.cards = []
        self.stack = stack
        self.is_dealer = False
        self.is_all_in = False
        self.strategy = strategy
        self.verbose = verbose
        self.name = name
        self.side_pot = 0
        self.contribution_in_this_pot = 0
        self.player_type = 'default'

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
            try:
                if self.is_all_in:
                    if self.verbose:
                        print(self.name + ' did nothing (all in)')
                    return Action('null')
            except IndexError:
                raise IndexError((b_round, actions))

        action = self.strategy(self, board, pot, actions, b_round, opponent_stack, opponent_side_pot, blinds=BLINDS, verbose=self.verbose)
        if self.stack - action.value <= 0:
            self.is_all_in = True
            if self.verbose:
                print(self.name + ' goes all in (' + str(self.stack) + ')')
            return Action('all in', value=self.stack)
        if self.verbose:
            print(self.name + ' ' + action.type + ' (' + str(action.value) + ')')
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


class NeuralFictitiousPlayer(Player):
    '''
    NFSP
    '''
    def __init__(self, pid, strategy, stack, name, is_training=True, learning_rate=1e-3, gamma=.95, verbose=False):
        super().__init__()

        self.cards = []
        self.stack = stack
        self.is_dealer = False
        self.is_all_in = False
        self.verbose = verbose
        self.side_pot = 0
        self.contribution_in_this_pot = 0

        self.id = pid
        self.name = name

        self.player_type = 'nfp'
        self.strategy = strategy
        self.Q_used = False
        self.is_training = is_training
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.Q_target = QNetwork(NUM_ACTIONS, NUM_HIDDEN_LAYERS, featurizer)
        # logically they should fall under each player
        # so we can do player.model.Q, player.model.pi
        # experience replay
        rl_conf = {'size': 1e+4,
                'learn_start': 100,
                'partition_num': 10,
                'total_step': 10000,
                'batch_size': 50
                }
        self.learn_start = rl_conf['learn_start']
        self.memory_rl = ReplayBufferManager(target='rl', **rl_conf)
        sl_conf = {
                   'size': 1e+4,
                   'batch_size': 50
                  }
        self.memory_sl = ReplayBufferManager(target='sl', **sl_conf)


    def play(self, state):
        '''
        TODO: check the output action dimension
        '''
        # check if chosen_action is invalid
        action, self.Q_used = self.strategy.choose_action(*state)
        return chosen_action

    def learn(self, global_step, episode_i, is_training=True):
        '''
        NSFP algorithm: learn on batch
        TODO: add a second player with Q and PI
        '''
        # TODO: set policy with
        if global_step > self.learn_start:
           self._learn_rl()
           self._learn_sl()

        if episode_i % self.target_update == 0:
            # sync target network periodically
            self.strategy.sync_target_network()

    def _learn_rl(self):
        # sample a minibatch of experiences
        gamma = Variable(t.Tensor([self.gamma]).float(), requires_grad=False)
        exps, imp_weights, ids = self.memory_rl.sample()
        states = create_state_var(exps[:, 0])
        actions = create_action_var(exps[:, 1])
        rewards = create_reward_var(exps[:, 2])
        next_states = create_state_var(exps[:, 3])
        if self.is_training:
            Q_targets = rewards + GAMMA * self.Q_target.forward(*next_states)[:, 0].squeeze()
            td_deltas = self.Q.train(states, Q_targets, imp_weights)
            self.memory_rl.update(ids, td_deltas)

    def _learn_sl(self):
       '''
       reservior sampling from M_sl
       '''
        if self.is_training:
            exps, ids = self.memory_sl.sample()
            states = create_state_var(exps[:, 0])
            actions = create_action_var(exps[:, 1])
            self.pi.train(states, actions)

    def remember(self, exp):
        self.memory_rl.store_experience(exp)
        if self.Q_used:
            # if action was chosen by e-greedy policy
            # exp should be just (s,a)
            simple_exp = {'s': exp['s'], 'a': exp['a']}
            self.memory_sl.store_experience(simple_exp)


