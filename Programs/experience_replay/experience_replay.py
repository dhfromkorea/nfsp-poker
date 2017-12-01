from experience_replay.proportional import ProportionalExperienceReplay
from experience_replay.rank_based import RankExperienceReplay
from experience_replay.reservoir import ReservoirExperienceReplay
import numpy as np
import math


class ReplayBufferManager:
    '''
    use only rank based type for now.
    TODO: support proportion based and nfsp
    see this example
    https://github.com/Damcy/cascadeLSTMDRL/blob/a6c502bc93197adb36adc8313cc925fdb12c08ee/agent/src/QLearner.py
    '''

    def __init__(self, target, config, learn_start, verbose=False):
        if not target in ['rl', 'sl']:
            raise Exception('Unsupported Memory Type', target)

        self.target = target
        # apply constraints on the config here
        if 'batch_size' in config:
            assert config['batch_size'] < learn_start, "You can't start learning before having \
            batch_size samples"
        if 'partiion_num' in config:
            assert config['batch_size'] < config['partition_num'], "You can't have partition \
            number smaller than batch size"
            assert config['partition_num'] < learn_start, "You can't start learning before \
            having partition_num samples"

        if self.target == 'rl':
            self.config = {'size': config.get('size', 2 ** 10),
                           # this is a game-level parameter
                           'learn_start': learn_start,
                           'partition_num': config.get('partition_num', 2 ** 4),
                           # when bias decay schedule ends
                           'total_step': config.get('total_step', 10 ** 9),
                           'batch_size': config.get('batch_size', 2 ** 5)
                           }

            dist_index = self.config['learn_start'] / self.config['size'] * self.config['partition_num']
            assert math.floor(dist_index) == math.ceil(dist_index), "Memory_RL initialization should be consistent with the assertion here"
            self._buffer = RankExperienceReplay(self.config)
        elif self.target == 'sl':
            self.config = {'size': config.get('size', 10 ** 6),
                           'learn_start': learn_start,
                           'batch_size': config.get('batch_size', 64)
                           }
            self._buffer = ReservoirExperienceReplay(self.config)
        else:
            raise Exception('Experience Replay target not supported')

        if True:
            print('experience replay set up')
            print(self.config)

        self.batch_size = config.get('batch_size', 64)
        self._last_step_buffer = None

    @staticmethod
    def make_exp_tuple(experience):
        return (experience['s'], experience['a'],
                experience['r'], experience['next_s'],
                experience['t'])

    def store_experience(self, experience):
        if self.target == 'sl':
            return self.store(experience)

        # store experience
        # note: timestep is needed to compute importance weights
        # check this paper: https://arxiv.org/pdf/1511.05952.pdf
        # @debug
        # if not self.is_last_step_buffer_empty and not experience['is_new_game']:
        if not self.is_last_step_buffer_empty:
            # update T_{t-1}
            self._last_step_buffer['next_s'] = experience['s']
            if experience['is_terminal']:
                self._last_step_buffer['r'] += experience['final_reward']
            # store T_{t-1} in a real buffer
            exp_tuple = ReplayBufferManager.make_exp_tuple(self._last_step_buffer)
            self.store(exp_tuple)

        # we flush empty temp buffer is a new episode starts
        if experience['is_terminal']:
            self._last_step_buffer = None
        else:
            # put T_{t} in a temp buffer
            self._last_step_buffer = experience

    @property
    def size(self):
        return self._buffer.record_size

    @property
    def is_last_step_buffer_empty(self):
        return self._last_step_buffer == None

    def store(self, exp_tuple):
        res = self._buffer.store(exp_tuple)
        if not res:
            raise Exception('failed to store', exp_tuple)

    def sample(self, global_step):
        '''
        params:
            global step: required to anneal the bias (beta)
        returns:
            exps: numpy array of (s, a, r, next_s, t)
                  whose dimension is batch_size x 5
            weights: importance weights to adjust for sampling bias
            exp_ids: experience ids required for updates later
        '''
        if self.target == 'rl':
            exps, imp_weights, exp_ids = self._buffer.sample(global_step)
            if exps == False:
                raise Exception('check learn start vs.')
            return self._batch_stack(exps), imp_weights, exp_ids
        else:
            exps = self._buffer.sample()
            return self._batch_stack(exps)

    def update(self, exp_ids, deltas):
        '''
        params:
            exp_ids: experience ids provided from sampling
            deltas: list of absolute td errors
        '''
        if self.target == 'rl':
            self._buffer.update_priority(exp_ids, deltas)

    def _batch_stack(self, exps):
        '''
        fix this...
        '''
        exps_batch = []
        num_features = 11
        # suboptimal performance
        # let's refactor later
        if self.target == 'rl':
            # TODO: refactor this
            states = exps[:, 0]
            actions = np.stack(exps[:, 1])
            rewards = exps[:, 2]
            next_states = exps[:, 3]
            time_steps = exps[:, 4]

            state_batch = [[] for _ in range(num_features)]
            for s in states:
                for i, feature in enumerate(s):
                    state_batch[i].append(feature)
            state_batch = [np.concatenate(s) for s in state_batch]

            next_state_batch = [[] for _ in range(num_features)]
            for s in next_states:
                for i, feature in enumerate(s):
                    next_state_batch[i].append(feature)
            next_state_batch = [np.concatenate(s) for s in next_state_batch]

            exps_batch.append(state_batch)
            exps_batch.append(actions)
            exps_batch.append(rewards)
            exps_batch.append(next_state_batch)
            exps_batch.append(time_steps)

        elif self.target == 'sl':
            states = [e[0] for e in exps]
            actions = [e[1] for e in exps]

            state_batch = [[] for _ in range(num_features)]
            for s in states:
                for i, feature in enumerate(s):
                    state_batch[i].append(feature)
            state_batch = [np.concatenate(s) for s in state_batch]
            exps_batch.append(state_batch)
            exps_batch.append(actions)
        else:
            raise Exception('Unsuported Experience Replay Target')
        return exps_batch
