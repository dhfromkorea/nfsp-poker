from experience_replay.proportional import ProportionalExperienceReplay
from experience_replay.rank_based import RankExperienceReplay
#from experience_replay.rank_based import RankExperienceReplay

SUPPORTED_PRIORITY_TYPES = {}
SUPPORTED_PRIORITY_TYPES['rank'] = RankExperienceReplay
SUPPORTED_PRIORITY_TYPES['proportion'] = ProportionalExperienceReplay
#SUPPORTED_PRIORITY_TYPES['nsfp'] = NsfpExperienceReplay

class ReplayBufferManager():
    def __init__(self, case='normal'):
        # T_{t-1} -> self._temp_buffer
        self._temp_buffer = None
        self.is_empty = True
        self.count = 0
        self._buffer_rl = ReplayBuffer()
        if case == 'nsfp':
            self._buffer_sl = ReplayBuffer()

    def make_exp_tuple(self, transition):
        return (transition['s'], transition['a'],
                transition.get('r', 0), transition.get('next_s', 'TERMINAL'),
                transition['t'])

    def store_transition(self, transition, buffer_type='rl'):
        '''
        buffer_type: 'rl' or 'sl'
        TODO: implement buffer_type
        '''
        # store transition
        # note: timestep is needed to compute importance weights
        # check this paper: https://arxiv.org/pdf/1511.05952.pdf
        if not self.is_empty and transition['is_new_game']:
            # don't take into account transitions overlapping two different games
            # update T_{t-1}
            import pdb;pdb.set_trace()
            self._temp_buffer['next_s'] = transition['s']
            if transition['s'] == 'TERMINAL':
                self._temp_buffer['r'] += transition['final_reward']
            # store T_{t-1} in a real buffer
            exp_tuple = make_exp_tuple(self._temp_buffer)
            replay_buffer.store(exp_tuple)
            self._count += 1
            self.is_empty = False

        # put T_{t} in a temp buffer
        self._temp_buffer = transition

    def size(self):
        return self.count



class ReplayBuffer:
    '''
    use only rank based type for now.
    TODO: support proportion based and nsfp
    see this example
    https://github.com/Damcy/cascadeLSTMDRL/blob/a6c502bc93197adb36adc8313cc925fdb12c08ee/agent/src/QLearner.py
    '''
    def __init__(self, priority_type='rank', size=500,
                 learn_start=10, partition_num=5,
                 total_step=100, batch_size=10):

        if priority_type not in SUPPORTED_PRIORITY_TYPES.keys():
            msg = 'unsupported replay buffer type: {}'.format(priority_strategy)
            raise Exception(msg)
        self.conf = {'size': size,
                'learn_start': learn_start,
                'partition_num': partition_num,
                'total_step': total_step,
                'batch_size': batch_size}

        self.buffer = SUPPORTED_PRIORITY_TYPES[priority_type](self.conf)


    def store(self, exp_tuple):
        '''
        experience is a tuple
        like(state_t, a, r, next_state, t)
        for nsfi M_sl
        like (state_t, a, t)
        TODO: not sure when we'd need t
        but let's keep it there
        '''
        self.buffer.store(exp_tuple)


    def sample(self, sample_size, global_step, require_importance_weights=False):
        '''
        global step is needed to decay the bias
        return:
            samples: sample = (exp_tuple, importance_weight, exp_id)
        '''
        samples = []
        for _ in range(sample_size):
            res = replay_buffer.sample(global_step)
            samples.append(res)
        return samples


    def update(self, exp_id, update_delta):
        '''
        update_delta: td error
        '''
        self.buffer.update_priority(exp_id, update_delta)
