'''Possible Experiments
1. NFSP RL agent against a totally random agent
2. NFSP RL agent against a mirror agent
3. NFSP RL agent against a lagged agent
4. NFSP RL agent against a simple DDQN(eta = 0) and the average policy agent(eta=1)
'''
import glob as g
import pickle
import evaluation.expt_utils as eu
import argparse

GAME_SCORE_HISTORY_PATH = 'data/game_score_history/'
PLAY_HISTORY_PATH = 'data/play_history/'
NEURAL_NETWORK_HISTORY_PATH = 'data/neural_network_history/'


def load_results(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='process configuration vars')
    parser.add_argument('--cuda', action='store_true', dest='cuda')
    parser.add_argument('--verbose', action='store_true', dest='verbose')
    parser.set_defaults(cuda=False)
    parser.set_defaults(verbose=False)
    args = parser.parse_args()
    cuda = args.cuda
    verbose = args.verbose

    results_dict = {}
    # results_dict['Random vs Random'] = eu.conduct_games('random', 'random', num_games = 100, mov_avg_window = 5)

    # example
    # caution: with partition 2**11, it will take minutes to initialize memory
    memory_rl_config = {
                   'size': 2 ** 17,
                   'partition_num': 2 ** 11,
                   'total_step': 10 ** 9,
                   'batch_size': 2 ** 5
                   }
    memory_sl_config = {
                   'size': 2 ** 15,
                   'batch_size': 2 ** 6
                   }
    results_dict['NFSP vs random'] = eu.conduct_games('NFSP', 'random',
                                                      learn_start=2**7,
                                                      num_games=10000,
                                                      eta_p1=.5,
                                                      mov_avg_window=100,
                                                      log_freq=100,
                                                      memory_rl_config=memory_rl_config,
                                                      memory_sl_config=memory_sl_config,
                                                      cuda=cuda,
                                                      verbose=verbose
                                                      )
    # eu.plot_results(results_dict)

    # pick the latest created file == results just created from the simulation above
    game_score_history_paths = g.glob(GAME_SCORE_HISTORY_PATH + '*')[-1]
    play_history_paths = g.glob(PLAY_HISTORY_PATH + '*')[-1]
    neural_network_history_paths = g.glob(NEURAL_NETWORK_HISTORY_PATH + '*')[-1]

    print('game history')
    print(load_results(game_score_history_paths))
    print('play history')
    for k, v in load_results(play_history_paths).items():
        print(k)
        print(v)
        print()
    print('neural network history')
    print(load_results(neural_network_history_paths))
