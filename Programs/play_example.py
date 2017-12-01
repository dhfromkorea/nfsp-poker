"""
This script allows to run poker simulations while training an RL agent

Note:
To make the problem simpler, some actions are impossible
For example, we forbid that the agents min-raises more than twice in a given betting round.
It doesn't lose a lot of generality anyway, since it represents most situations. It has the advantage of greatly reducing the number of possible
actions per betting round

"""

from game.simulator import Simulator

# FEATURIZER_NAME = 'c11_h50xf10_model9'
FEATURIZER_NAME = 'card_featurizer1.50-10.model.pytorch'
SAVED_FEATURIZER_PATH = 'data/hand_eval/best_models/' + FEATURIZER_NAME
#SAVED_FEATURIZER_PATH = 'data/hand_eval/2017_11_25/saved_models/' + 'card_featurizer1.50-10.model.pytorch'

if __name__ == '__main__':
    # TODO: arg parser
    cuda = False
    verbose = False
    simulator = Simulator(verbose=verbose, featurizer_path=SAVED_FEATURIZER_PATH, cuda=cuda,
                          p1_strategy='NFSP', p2_strategy='NFSP')
    simulator.start()
