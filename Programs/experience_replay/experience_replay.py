from experience_replay.proportional import ProportionalExperienceReplay
from experience_replay.rank_based import RankExperienceReplay
#from experience_replay.reservoir import ReservoirExperienceReplay
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


        if priority_type not in SUPPORTED_PRIORITY_TYPES.keys():
            msg = 'unsupported replay buffer type: {}'.format(priority_strategy)
            raise Exception(msg)

        if target='rl':
            self.conf = {'size': size,
                    'learn_start': learn_start,
                    'partition_num': partition_num,
                    'total_step': total_step,
                    'batch_size': batch_size}

            self._buffer = SUPPORTED_PRIORITY_TYPES[priority_type](self.conf)
        elif target='sl':
            pass
            #self._buffer = ReservoirExperienceReplay
        else:
            raise Exception('Experience Replay target not supported')


        self._last_step_buffer = None

    @staticmethod
    def make_exp_tuple(experience):
        return (experience['s'], experience['a'],
                experience['r'], experience['next_s'],
                experience['t'])


    def store_experience(self, experience):
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
        return self._buffer.sample(global_step)


    def update(self, exp_ids, deltas):
        '''
        params:
            exp_ids: experience ids provided from sampling
            deltas: list of absolute td errors
        '''
        self._buffer.update_priority(exp_ids, deltas)

