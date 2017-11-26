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
TARGET_NETWORK_UPDATE_PERIOD = 300 # every 300 episodes
ANTICIPATORY_PARAMETER = 0.1
EPSILON = 0.01
NUM_HIDDEN_LAYERS = 10
NUM_ACTIONS = 14
GAMMA_VAL = 0.95

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
        # experience replay
        rl_conf = {'size': 1000,
                'learn_start': 100,
                'partition_num': 10,
                'total_step': 10000,
                'batch_size': 10
                }
        self.memory_rl = ReplayBufferManager(target='rl', **rl_conf)

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
    def __init__(self, pid, name):
        self.id = pid
        self.name = name
        # logically they should fall under each player
        # so we can do player.model.Q, player.model.pi
        self.Q = QNetwork(NUM_ACTIONS, NUM_HIDDEN_LAYERS)
        self.Q_target = QNetwork(NUM_ACTIONS, NUM_HIDDEN_LAYERS)
        self.eta = ANTICIPATORY_PARAMETER
        self.target_update = TARGET_NETWORK_UPDATE_PERIOD
        self.epsilon = EPSILON
        self.pi = PiNetwork(NUM_ACTIONS, NUM_HIDDEN_LAYERS)
        self.player_type = 'nfp'

        # experience replay
        rl_conf = {'size': 1000,
                'learn_start': 100,
                'partition_num': 10,
                'total_step': 10000,
                'batch_size': 10
                }
        self.memory_rl = ReplayBufferManager(target='rl', **rl_conf)
        sl_conf = {'size': 1000,
                'learn_start': 100,
                'partition_num': 10,
                'total_step': 10000,
                'batch_size': 10
                }
        self.memory_sl = ReplayBufferManager(target='sl', **sl_conf)


    def act(self, state):
        '''
        TODO: check the output action dimension
        '''
        if self.eta > np.random.rand():
            # use epsilon-greey policy
            if self.epsilon > np.random.rand():
                #choose_random_actions
                chosen_action = 0
            else:
                q_vals = self.Q.forward(*state)[0].squeeze()
                # TODO: tiebreaking
                chosen_action = np.argmax(q_vals)
                # encode chosen_action in the expected dimension
        else:
            # use average policy
            q_vals = self.pi.forward(*state)[0].squeeze()
            chosen_action = np.argmax(q_vals)

        # check if chosen_action is invalid
        return chosen_action

    def learn(self, episode_i, is_training=True):
        '''
        NSFP algorithm: learn on batch
        TODO: add a second player with Q and PI
        '''
        # TODO: set policy with
        GAMMA = Variable(t.Tensor([GAMMA_VAL]).float(), requires_grad=False)
        # TODO: add another network for NSFP and M_sl
        # turns out M_SL does not use PER (it uses Reservoir Sampling (Vitter, 1985)
        # I will implement this soon
        # we start learning after LEARN_START (see params to ReplayBuffer)
        if global_step > 101:
           self._learn_rl()
           self._learn_sl()

        if  episode_i % self.target_update == 0:
            # sync target network periodically
            self._copy_model(self.Q, self.Q_target)

    def _learn_rl(self):
        # sample a minibatch of experiences
        exps, imp_weights, ids = self.memory_rl.sample(global_step=global_step)
        states = create_state_var(exps[:, 0])
        actions = create_action_var(exps[:, 1])
        rewards = create_reward_var(exps[:, 2])
        next_states = create_state_var(exps[:, 3])
        if is_training:
            targets = rewards + GAMMA * self.Q_target.forward(*next_states)[:, 0].squeeze()
            td_deltas = self.Q.train(states, actions, targets, imp_weights)
            self.memory_rl.update(ids, td_deltas)

    def _learn_sl(self):
       '''
       reservior sampling from M_sl
       '''
       pass

    def remember(self, exp):
        self.memory_rl.store_experience(exp)
        # if action was chosen by e-greedy policy
        # exp should be just (s,a)
        #self.memory_sl.store_experience(exp)


    def _copy_model(self, from_model, to_model):
        '''
        create a fixed target network
        copy weights and memorys
        '''
        to_model.load_state_dict(from_model.state_dict())
