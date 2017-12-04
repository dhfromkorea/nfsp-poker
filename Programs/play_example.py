"""
This script allows to run poker simulations while training an RL agent

Note:
To make the problem simpler, some actions are impossible
For example, we forbid that the agents min-raises more than twice in a given betting round.
It doesn't lose a lot of generality anyway, since it represents most situations. It has the advantage of greatly reducing the number of possible
actions per betting round

"""

from game.simulator import Simulator
import argparse

FEATURIZER_NAME = 'card_featurizer1.50-10.model.pytorch'
SAVED_FEATURIZER_PATH = 'data/hand_eval/best_models/' + FEATURIZER_NAME

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='process configuration vars')
    parser.add_argument('--cuda', action='store_true', dest='cuda')
    parser.add_argument('--verbose', action='store_true', dest='verbose')
    parser.set_defaults(cuda=False)
    parser.set_defaults(verbose=False)
    args = parser.parse_args()

    cuda = False
    p1_strategy = 'NFSP'
    p2_strategy = 'mirror'
    learn_start = 2**7
    eta_p1 = .75
    eta_p2 = .1
    eps = .1
    gamma = .99
    learning_rate = 1e-4
    target_Q_update_freq = 200

    memory_rl_config = {
        # 'size': 2 ** 17,
        'size': 2**10,
        # 'partition_num': 2 ** 11,
        'partition_num': 2**7,
        # 'total_step': 10 ** 9,
        'total_step': 10**9,
        # 'batch_size': 2 ** 5
        'batch_size': 2**5,
    }
    memory_sl_config = {
        'size': 2 ** 15,
        'batch_size': 2 ** 6
    }

    simulator = Simulator(p1_strategy=p1_strategy,
                          p2_strategy=p2_strategy,
                          learn_start=learn_start,
                          cuda=cuda,
                          eta_p1=eta_p1,
                          eta_p2=eta_p2,
                          gamma=gamma,
                          eps=eps,
                          learning_rate=learning_rate,
                          target_Q_update_freq=target_Q_update_freq,
                          memory_rl_config=memory_rl_config,
                          memory_sl_config=memory_sl_config,
                          verbose=True,
                          log_freq=100)
    simulator.start()
