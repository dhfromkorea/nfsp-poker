from torch.autograd import Variable
import torch as t
import numpy as np

from experience_replay.experience_replay import ReplayBufferManager
from game.game_utils import Action
from game.utils import variable
from game.state import build_state, create_state_variable_batch, create_state_vars_batch
from game.action import create_action_variable_batch
from game.reward import create_reward_variable_batch
from game.config import BLINDS

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

    def __init__(self, pid, strategy, stack, name,
                 learn_start, memory_rl_conf={}, memory_sl_conf={},
                 is_training=True, learning_rate=1e-3, gamma=.95,
                 target_update=1000, verbose=False, cuda=False):
        # we may not need this inheritance
        super().__init__(pid, strategy, stack)
        self.cuda = cuda

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
        self.is_Q_used = False
        self.is_training = is_training
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.target_update = target_update

        # logically they should fall under each player
        # so we can do player.model.Q, player.model.pi
        # experience replay
        self.learn_start = learn_start
        self.memory_rl = ReplayBufferManager(target='rl', config=memory_rl_conf, learn_start=learn_start)
        self.memory_sl = ReplayBufferManager(target='sl', config=memory_sl_conf, learn_start=learn_start)
        self.memory_count = 0

    def play(self, board, pot, actions, b_round, opponent_stack, opponent_side_pot, blinds):
        '''
        TODO: check the output action dimension
        '''

        action, self.is_Q_used = self.strategy.choose_action(self, board, pot, actions, b_round, opponent_stack, opponent_side_pot, blinds)
        return action

    def learn(self, global_step, episode_i, is_training=True):
        '''
        NSFP algorithm: learn on batch
        TODO: add a second player with Q and PI
        '''
        # TODO: set policy with
        if self._is_ready_to_learn(global_step):
            self._learn_rl(global_step)
            self._learn_sl(global_step)

        if episode_i % self.target_update == 0:
            # sync target network periodically
            self.strategy.sync_target_network()


    def _is_ready_to_learn(self, global_step):
        # with some safety padding for off by one error
        hot_fix = self.memory_rl._buffer.record_size > self.memory_rl._buffer.batch_size
        return global_step > (self.learn_start) and hot_fix


    def _learn_rl(self, global_step):
        '''
        '''
        # sample a minibatch of experiences
        gamma = Variable(t.Tensor([self.gamma]).float(), requires_grad=False)
        exps, imp_weights, ids = self.memory_rl.sample(global_step)
        state_vars = [variable(s) for s in exps[0]]
        action_vars = variable(exps[1])
        imp_weights = variable(imp_weights)
        rewards = variable(exps[2].astype(np.float32))
        next_state_vars = [variable(s) for s in exps[3]]

        if self.is_training:
            Q_targets = rewards + gamma * t.max(self.strategy._target_Q.forward(*next_state_vars), 1)[0]
            td_deltas = self.strategy._Q.learn(state_vars, Q_targets, imp_weights)
            self.memory_rl.update(ids, td_deltas.data.numpy())

    def _learn_sl(self, global_step):
        '''
       reservior sampling from M_sl
       '''
        if self.is_training:
            exps = self.memory_sl.sample(global_step)
            state_vars = [variable(s) for s in exps[0]]
            # 4 x 11 each column is torch variable
            action_vars = variable(exps[1])
            self.strategy._pi.learn(state_vars, action_vars)

    def remember(self, exp):
        self.memory_count += 1
        print(self.name, ' remember count', self.memory_count)
        self.memory_rl.store_experience(exp)
        if self.is_Q_used:
            # if action was chosen by e-greedy policy
            # exp should be just (s,a)
            simple_exp = (exp['s'], exp['a'])
            self.memory_sl.store_experience(simple_exp)
