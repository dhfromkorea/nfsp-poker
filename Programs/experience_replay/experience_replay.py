from experience_replay.proportional import ProportionalExperienceReplay
from experience_replay.rank_based import RankExperienceReplay
from experience_replay.reservoir import ReservoirExperienceReplay
import numpy as np

SUPPORTED_PRIORITY_TYPES = {}
SUPPORTED_PRIORITY_TYPES['rank'] = RankExperienceReplay
SUPPORTED_PRIORITY_TYPES['proportion'] = ProportionalExperienceReplay


class ReplayBufferManager:
    '''
    use only rank based type for now.
    TODO: support proportion based and nsfp
    see this example
    https://github.com/Damcy/cascadeLSTMDRL/blob/a6c502bc93197adb36adc8313cc925fdb12c08ee/agent/src/QLearner.py
    '''
    def __init__(self, target='rl', priority_type='rank', size=100,
                 learn_start=10, partition_num=5,
                 total_step=200, batch_size=5):

        self.target = target

        if priority_type not in SUPPORTED_PRIORITY_TYPES.keys():
            msg = 'unsupported replay buffer type: {}'.format(priority_strategy)
            raise Exception(msg)

        if self.target == 'rl':
            self.conf = {'size': size,
                    'learn_start': learn_start,
                    'partition_num': partition_num,
                    'total_step': total_step,
                    'batch_size': batch_size}

            self._buffer = SUPPORTED_PRIORITY_TYPES[priority_type](self.conf)
        elif self.target == 'sl':
            self.conf = {'size': size,
                         'batch_size': batch_size}
            self._buffer = ReservoirExperienceReplay(self.conf)
        else:
            raise Exception('Experience Replay target not supported')


        self.batch_size = batch_size
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
        if not self.is_last_step_buffer_empty and not experience['is_new_game']:
            # update T_{t-1}
            self._last_step_buffer['next_s'] = experience['s']
            if experience['s'] == 'TERMINAL':
                self._last_step_buffer['r'] += experience['final_reward']
            # store T_{t-1} in a real buffer
            exp_tuple = ReplayBufferManager.make_exp_tuple(self._last_step_buffer)
            self.store(exp_tuple)

        # we flush empty temp buffer is a new episode starts
        if experience['s'] == 'TERMINAL':
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
        exps_batch = []
        # suboptimal performance
        # let's refactor later
        if self.target == 'rl':
            # TODO: refactor this
            states = exps[:, 0]
            actions = np.stack(exps[:, 1])
            rewards = exps[:, 2]
            next_states = exps[:, 3]
            time_steps = exps[:, 4]
            num_features = 11
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
            states = exps[:, 0]
            actions = exps[:, 1]
            for i in range(self.batch_size):
                # state
                # hack: needed to drop 1 extra dim
                states[i][0] = states[i][0].squeeze()
                s_batch[i] = np.array(states[i])
                # action
            exps_batch.append(s_batch)
            exps_batch.append(actions)
        else:
            raise Exception('Unsuported Experience Replay Target')
        return exps_batch
