from torch.autograd import Variable
import torch as t
import numpy as np
from timeit import default_timer as timer
import time

from experience_replay.experience_replay import ReplayBufferManager
from game.game_utils import Action, bucket_encode_actions
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
        self.has_played = False
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

    def play(self, board, pot, actions, b_round, opponent_stack, opponent_side_pot, blinds, episode_idx):
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

    def __init__(self, pid,
                 strategy,
                 stack,
                 name,
                 learn_start,
                 learning_freq,
                 gamma,
                 target_Q_update_freq,
                 memory_rl_config={},
                 memory_sl_config={},
                 tensorboard=None,
                 is_training=True,
                 verbose=False,
                 cuda=False):
        # we may not need this inheritance
        super().__init__(pid, strategy, stack)
        self.cuda = cuda
        self.verbose = verbose

        self.cards = []
        self.stack = stack
        self.is_dealer = False
        self.is_all_in = False
        self.has_played = False
        self.verbose = verbose
        self.side_pot = 0
        self.contribution_in_this_pot = 0

        self.id = pid
        self.name = name

        self.player_type = 'nfsp'
        self.strategy = strategy
        self.is_Q_used = False
        self.is_training = is_training
        self.gamma = gamma
        self.target_update = target_Q_update_freq
        self.learning_freq = learning_freq
        self.tensorboard = tensorboard

        # logically they should fall under each player
        # so we can do player.model.Q, player.model.pi
        # experience replay
        self.learn_start = learn_start
        self.memory_rl = ReplayBufferManager(target='rl', config=memory_rl_config, learn_start=learn_start)
        self.memory_sl = ReplayBufferManager(target='sl', config=memory_sl_config, learn_start=learn_start)

    def play(self, board, pot, actions, b_round, opponent_stack, opponent_side_pot, blinds,
             episode_idx):
        """
        TODO: check the output action dimension
        """
        action, self.is_Q_used = self.strategy.choose_action(self, board, pot,
                                                             actions, b_round,
                                                             opponent_stack, opponent_side_pot,
                                                             blinds, episode_idx, for_play=True)

        return action

    def learn(self, global_step, episode_idx, is_training=True):
        """
        NSFP algorithm: learn on batch
        TODO: add a second player with Q and PI
        """
        if self.verbose:
            print('global step', global_step)
            print('Record Size of M_RL', self.memory_rl._buffer.record_size)
            print('Record Size of M_SL', self.memory_sl._buffer.record_size)

        if episode_idx % self.learning_freq == 0:
            # learn only every X number of episodes
            # episode_i increments one by one
            if self._is_ready_to_learn_RL(global_step):
                self._learn_rl(global_step)

            if self._is_ready_to_learn_SL(global_step):
                self._learn_sl(global_step)

        record_size_rl = self.memory_rl._buffer.record_size
        record_size_sl = self.memory_sl._buffer.record_size
        msg = 'pid: {} record size for RL:{} should be larger than SL: {}'.format(self.id, record_size_rl, record_size_sl)
        # assert (record_size_rl + 1) >= record_size_sl, msg

        if episode_idx % self.target_update == 0:
            if self.verbose:
                print('sync target network periodically')
            self.strategy.sync_target_network()

    def _is_ready_to_learn_RL(self, global_step):
        record_size = self.memory_rl._buffer.record_size
        batch_size = self.memory_rl.batch_size
        return record_size >= self.learn_start and record_size >= batch_size

    def _is_ready_to_learn_SL(self, global_step):
        record_size = self.memory_sl._buffer.record_size
        batch_size = self.memory_sl.batch_size
        return record_size >= batch_size

    def _learn_rl(self, global_step):
        # sample a minibatch of experiences
        # gamma = Variable(t.Tensor([self.gamma]).float(), requires_grad=False)
        gamma = variable([self.gamma], cuda=self.cuda)
        exps, imp_weights, ids = self.memory_rl.sample(global_step)
        # how many of the samples in a batch are showdowns or all-ins
        state_vars = [variable(s, cuda=self.cuda) for s in exps[0]]
        action_vars = variable(exps[1], cuda=self.cuda)
        imp_weights = variable(imp_weights, cuda=self.cuda)
        rewards = variable(exps[2].astype(np.float32), cuda=self.cuda)
        next_state_vars = [variable(s, cuda=self.cuda) for s in exps[3]]
       # state_hashes = exps[5]

        if self.tensorboard is not None:
            actions = bucket_encode_actions(action_vars, cuda=self.cuda)
            for a in actions.data.cpu().numpy():
                self.tensorboard.add_scalar_value('M_RL_sampled_actions', int(a), time.time())
            for r in exps[2]:
                self.tensorboard.add_scalar_value('M_RL_sampled_rewards', int(r), time.time())
#            for h in state_hashes:
#                self.tensorboard.add_scalar_value('M_RL_sampled_states', int(h), time.time())


        if self.is_training:
            Q_targets = rewards + gamma * t.max(self.strategy._target_Q.forward(*next_state_vars), 1)[0]

            if self.verbose:
                start = timer()
            td_deltas = self.strategy._Q.learn(state_vars, action_vars, Q_targets, imp_weights)
            if self.verbose:
                print('backward pass of Q network took ', timer() - start)
            self.memory_rl.update(ids, td_deltas.data.cpu().numpy())

    def _learn_sl(self, global_step):
        """
        reservoir sampling from M_sl
        """
        if self.is_training:
            exps = self.memory_sl.sample(global_step)
            state_vars = [variable(s, cuda=self.cuda) for s in exps[0]]
            # 4 x 11 each column is torch variable
            action_vars = variable(exps[1], cuda=self.cuda)
            #state_hashes = exps[2]
            if self.tensorboard is not None:
                actions= bucket_encode_actions(action_vars, cuda=self.cuda)
                for a in actions.data.cpu().numpy():
                    self.tensorboard.add_scalar_value('M_SL_sampled_actions', int(a), time.time())
#                for h in state_hashes:
#                    self.tensorboard.add_scalar_value('M_SL_sampled_states', int(h), time.time())

            if self.verbose:
                start = timer()
            self.strategy._pi.learn(state_vars, action_vars)
            if self.verbose:
                print('backward pass of pi network took ', timer() - start)

    def remember(self, exp):
        self.memory_rl.store_experience(exp)
        if self.is_Q_used and not exp['is_terminal']:
            # if action was chosen by e-greedy policy
            # exp should be just (s,a)
            simple_exp = (exp['s'], exp['a'])
            self.memory_sl.store_experience(simple_exp)
