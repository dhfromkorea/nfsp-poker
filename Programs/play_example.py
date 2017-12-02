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

    simulator = Simulator(verbose=args.verbose,
                          featurizer_path=SAVED_FEATURIZER_PATH,
                          cuda=args.cuda,
                          p1_strategy='NFSP',
                          p2_strategy='random')
    simulator.start()
