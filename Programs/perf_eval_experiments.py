
'''Possible Experiments
1. NFSP RL agent against a totally random agent
2. NFSP RL agent against a mirror agent
3. NFSP RL agent against a lagged agent
4. NFSP RL agent against a simple DDQN(eta = 0) and the average policy agent(eta=1)
'''
import glob as g
import pickle
import evaluation.expt_utils as eu


GAME_SCORE_HISTORY_PATH = 'data/game_score_history/'
PLAY_HISTORY_PATH = 'data/play_history/'
NEURAL_NETWORK_HISTORY_PATH = 'data/neural_network_history/'

def load_results(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


if __name__ == '__main__':

    #Example
    results_dict = {}
    #results_dict['Random vs Random'] = eu.conduct_games('random', 'random', num_games = 100, mov_avg_window = 5)
    results_dict['NFSP vs NFSP'] = eu.conduct_games('NFSP','NFSP',
                                                    learn_start = 64,
                                                    num_games = 20,
                                                    mov_avg_window = 2)
    eu.plot_results(results_dict)

    game_score_history_paths = g.glob(GAME_SCORE_HISTORY_PATH +'*')[-1]
    play_history_paths = g.glob(PLAY_HISTORY_PATH +'*')[-1]
    neural_network_history_paths = g.glob(NEURAL_NETWORK_HISTORY_PATH +'*')[-1]

    print('game history')
    print(load_results(game_score_history_paths))
    print('play history')
    print(load_results(play_history_paths))
    print('neural network history')
    print(load_results(neural_network_history_paths))
'''
Things to log
1. game results (done)
2. play history
key: global_step
values:
   states
   actions
   game-level info (dealer, to_play, big_blind)
3. network log
- Q value
- pi value (optional)
'''

