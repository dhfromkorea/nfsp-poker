import sys
import math
import random
from collections import deque
import pickle

RESERVOIR_ER = ''

class ExperienceReplayStoreError(Exception):
    pass

class ReservoirExperienceReplay():
    # TODO: save the experience in a pickle
    def __init__(self, conf):
        self.size = conf['size']
        self.batch_size = conf['batch_size']
        self._buffer = deque(maxlen=self.size)

    @property
    def buffer(self):
        return self._buffer


    @property
    def record_size(self):
        return len(self._buffer)


    @record_size.setter
    def record_size(self, val):
        self._record_size = val

    def store(self, experience):
        '''
        experience is a tuple of (s, a, exp_id)
        where exp_id is record index
        '''
        # if failed
        exp_id = self.record_size
        experience += (exp_id,)
        try:
            if not self.is_full:
                self.buffer.append(experience)
            else:
                self.buffer.popleft()
                self.buffer.append(experience)
            self.record_size = self.record_size + 1
            return True
        except:
            raise ExperienceReplayStoreError
            return False

    def sample(self):
        '''
        '''
        try:
            return random.sample(self.buffer, self.batch_size)
        except ValueError:
            print('Not enough data to sample from the buffer')
            return None

    def update(self):
        '''
        there's no updating yet
        '''
        pass

    @property
    def is_full(self):
        return self.record_size >= self.size


