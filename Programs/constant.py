from game.config import BLINDS

# paths
SAVED_FEATURIZER_PATH = 'data/hand_eval/best_models/card_featurizer1.50-10.model.pytorch'
GAME_SCORE_HISTORY_PATH = 'data/game_score_history/'
PLAY_HISTORY_PATH = 'data/play_history/'
NEURAL_NETWORK_HISTORY_PATH = 'data/neural_network_history/'
NEURAL_NETWORK_LOSS_PATH = 'data/neural_network_history/loss/'
EXPERIMENT_PATH = 'data/tensorboard/'

# change the variable here
MODEL_SAVEPATH = 'data/neural_network_history/models/'
NUM_GAMES = 66000
MODEL_PATH_Q_P1 = 'data/neural_network_history/best_models/q_p1_{}g.pt'.format(NUM_GAMES)
MODEL_PATH_PI_P1 = 'data/neural_network_history/best_models/pi_p1_{}g.pt'.format(NUM_GAMES)
MODEL_PATH_Q_P2 = 'data/neural_network_history/best_models/q_p2_{}g.pt'.format(NUM_GAMES)
MODEL_PATH_PI_P2 = 'data/neural_network_history/best_models/pi_p2_{}g.pt'.format(NUM_GAMES)

# define game constants here
INITIAL_MONEY = 100 * BLINDS[0]
NUM_ROUNDS = 4  # pre, flop, turn, river
NUM_HIDDEN_LAYERS = 50
NUM_ACTIONS = 16
