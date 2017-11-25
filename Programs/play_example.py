"""
This script allows to run poker simulations while training an RL agent

Note:
To make the problem simpler, some actions are impossible
For example, we forbid that the agents min-raises more than twice in a given betting round.
It doesn't lose a lot of generality anyway, since it represents most situations. It has the advantage of greatly reducing the number of possible
actions per betting round

"""

from game.simulator import Simulator
from models.featurizer import FeaturizerManager

if __name__ == '__main__':
    #simulator = Simulator(verbose=True)
    #simulator.start()
    fm = FeaturizerManager(50, 10)
    fm.train()
