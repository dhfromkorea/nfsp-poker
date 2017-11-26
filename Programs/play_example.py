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

# change the model path to load the right one
MODEL_NAME = 'c11_h50xf10_model2'
SAVED_MODEL_PATH = 'data/hand_eval/2017_11_25/saved_models/' + MODEL_NAME

if __name__ == '__main__':
    # TODO: arg parser
    cuda = True
    #simulator = Simulator(verbose=True)
    #simulator.start()
    fm = FeaturizerManager(50, 10, cuda=cuda)
    fm.load_model(SAVED_MODEL_PATH)
    fm.train_featurizer1()
    
