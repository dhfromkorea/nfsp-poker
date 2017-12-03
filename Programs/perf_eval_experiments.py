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
import game

GAME_SCORE_HISTORY_PATH = 'data/game_score_history/'
PLAY_HISTORY_PATH = 'data/play_history/'
NEURAL_NETWORK_HISTORY_PATH = 'data/neural_network_history/'
NEURAL_NETWORK_LOSS_PATH = 'data/neural_network_history/loss/'

def load_results(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def get_arg_parser():
    parser = argparse.ArgumentParser(description='process configuration vars')
    # dev level
    parser.add_argument('-c', '--cuda', action='store_true', dest='cuda')
    parser.set_defaults(cuda=False)
    parser.add_argument('-v', '--verbose', action='store_true', dest='verbose')
    parser.set_defaults(verbose=False)
    parser.add_argument('--mov_avg_window', default=100, type=int, dest='mov_avg_window',
                        help='moving average for game results')
    # game/experiment
    parser.add_argument('-ng', '--num_games', default=10000, type=int, dest='num_games',
                        help='number of games to simulate')
    parser.add_argument('-lf', '--log_freq', default=100, type=int, dest='log_freq', help='log game results frequency')
    parser.add_argument('-ls', '--learn_start', default=2**7, type=int, dest='learn_start',
                        help='starting point for training networks')
    # experience replay
    parser.add_argument('-bs', '--batch_size', default=2**5, type=int, dest='batch_size',
                        help='batch size of Memory RL')
    parser.add_argument('-bfs', '--buffer_size', default=2**17, type=int, dest='buffer_size',
                        help='buffer size of Memory RL')
    parser.add_argument('-np', '--num_partitions', default=2**11, type=int, dest='num_partitions',
                        help='number of partitions to Memory RL')
    parser.add_argument('-ts', '--total_steps', default=10**9, type=int, dest='total_steps',
                        help='total steps to Memory RL')
    # define neural network parameters here

    return parser

if __name__ == '__main__':
    '''
    If you want to play around with the hyperparameters, you can do:

    (e.g.) python perf_eval_experiments.py -v -c -ng 100 -ls 128 -bs 4 -bfs 128 -np 16

    do: python script.py --flag value --another_flat value

    change the defaults above, if you don't want to pass flags
    '''
    args = get_arg_parser().parse_args()
    cuda = args.cuda
    verbose = args.verbose
    log_freq = args.log_freq
    num_games = args.num_games
    mov_avg_window = args.mov_avg_window
    learn_start = args.learn_start
    batch_size = args.batch_size
    buffer_size = args.buffer_size
    num_partitions = args.num_partitions
    total_steps = args.total_steps

    results_dict = {}
    # results_dict['Random vs Random'] = eu.conduct_games('random', 'random', num_games = 100, mov_avg_window = 5)

    memory_rl_config = {
                   #'size': 2 ** 17,
                   'size': buffer_size,
                   #'partition_num': 2 ** 11,
                   'partition_num': num_partitions,
                   #'total_step': 10 ** 9,
                   'total_step': total_steps,
                   #'batch_size': 2 ** 5
                   'batch_size': batch_size,
                   }
    memory_sl_config = {
                   'size': 2 ** 15,
                   'batch_size': 2 ** 6
                   }
    results_dict['NFSP vs random'] = eu.conduct_games('NFSP', 'random',
                                                      learn_start=learn_start,
                                                      num_games=num_games,
                                                      mov_avg_window=mov_avg_window,
                                                      log_freq=log_freq,
                                                      memory_rl_config=memory_rl_config,
                                                      memory_sl_config=memory_sl_config,
                                                      cuda=cuda,
                                                      verbose=verbose
                                                      )

    # pick the latest created file == results just created from the simulation above
    game_score_history_paths = g.glob(GAME_SCORE_HISTORY_PATH + '*')[-1]
    play_history_paths = g.glob(PLAY_HISTORY_PATH + '*')[-1]
    neural_network_history_paths = g.glob(NEURAL_NETWORK_HISTORY_PATH + '*')[-1]
    neural_network_loss_paths = g.glob(NEURAL_NETWORK_LOSS_PATH + '*')[-1]

    #eu.plot_results(results_dict)
    print('game history')
    print(load_results(game_score_history_paths))
    print('play history')
    for k, v in load_results(play_history_paths).items():
        print(k)
        print(v)
        print('')
    print('neural network history')
    print(load_results(neural_network_history_paths))
    print(load_results(neural_network_loss_paths))
