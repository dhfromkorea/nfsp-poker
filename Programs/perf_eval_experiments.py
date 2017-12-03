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

    parser.add_argument('-ss', '--skip_simulation', action='store_true',
                        dest='skip_simulation', help='show only the latest results without simulation')
    parser.set_defaults(skip_simulation=False)
    # game/experiment
    parser.add_argument('-ng', '--num_games', default=10000, type=int, dest='num_games',
                        help='number of games to simulate')
    parser.add_argument('-lf', '--log_freq', default=100, type=int, dest='log_freq', help='log game results frequency')
    parser.add_argument('-ls', '--learn_start', default=2 ** 7, type=int, dest='learn_start',
                        help='starting point for training networks')
    # experience replay
    parser.add_argument('-bs', '--batch_size', default=2 ** 5, type=int, dest='batch_size',
                        help='batch size of Memory RL')
    parser.add_argument('-bfs', '--buffer_size', default=2 ** 17, type=int, dest='buffer_size',
                        help='buffer size of Memory RL')
    parser.add_argument('-np', '--num_partitions', default=2 ** 11, type=int, dest='num_partitions',
                        help='number of partitions to Memory RL')
    parser.add_argument('-ts', '--total_steps', default=10 ** 9, type=int, dest='total_steps',
                        help='total steps to Memory RL')
    # define neural network parameters here

    parser.add_argument('-eps', '--epsilon', default=0.1, type=float, dest='eps',
                        help='eps')
    parser.add_argument('-g', '--gamma', default=0.95, type=float, dest='gamma',
                        help='gamma')
    parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float, dest='learning_rate',
                        help='learning rate')
    parser.add_argument('-tf', '--target_Q_update_freq', default=500, type=int,
                        dest='target_Q_update_freq', help='update target Q every X number of episodes')
    parser.add_argument('-ep1', '--eta_p1', default=0.75, type=float, dest='eta_p1',
                        help='eta for player 1')
    parser.add_argument('-ep2', '--eta_p2', default=0.5, type=float, dest='eta_p2',
                        help='eta for player 2')
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
    eta_p1 = args.eta_p1
    eta_p2 = args.eta_p2
    skip_simulation = args.skip_simulation
    eps = args.eps
    gamma = args.gamma
    learning_rate = args.learning_rate
    target_Q_update_freq = args.target_Q_update_freq

    results_dict = {}
    # results_dict['Random vs Random'] = eu.conduct_games('random', 'random', num_games = 100, mov_avg_window = 5)

    if not skip_simulation:
        # sometimes we want to skip simulation and view only the latest simulation results
        memory_rl_config = {
            # 'size': 2 ** 17,
            'size': buffer_size,
            # 'partition_num': 2 ** 11,
            'partition_num': num_partitions,
            # 'total_step': 10 ** 9,
            'total_step': total_steps,
            # 'batch_size': 2 ** 5
            'batch_size': batch_size,
        }
        memory_sl_config = {
            'size': 2 ** 15,
            'batch_size': 2 ** 6
        }
        results_dict['NFSP vs random'] = eu.conduct_games('NFSP', 'random',
                                                          # 2 ** 7
                                                          learn_start=learn_start,
                                                          # 10000
                                                          num_games=num_games,
                                                          # 100
                                                          mov_avg_window=mov_avg_window,
                                                          # 100
                                                          log_freq=log_freq,
                                                          # default for eta_p1 =0.5
                                                          eta_p1=eta_p1,
                                                          # default for eta_p1 =0.5
                                                          eta_p2=eta_p2,
                                                          # default 0.1
                                                          eps=eps,
                                                          # default 0.95
                                                          gamma=gamma,
                                                          # default 1e-3
                                                          learning_rate=learning_rate,
                                                          # default 100 episodes
                                                          target_Q_update_freq=target_Q_update_freq,
                                                          memory_rl_config=memory_rl_config,
                                                          memory_sl_config=memory_sl_config,
                                                          # default false
                                                          cuda=cuda,
                                                          # default false
                                                          verbose=verbose
                                                          )

    # pick the latest created file == results just created from the simulation above
    game_score_history_paths = g.glob(GAME_SCORE_HISTORY_PATH + '*')[-1]
    play_history_paths = g.glob(PLAY_HISTORY_PATH + '*')[-1]
    neural_network_history_paths = g.glob(NEURAL_NETWORK_HISTORY_PATH + '*')[-1]
    neural_network_loss_paths = g.glob(NEURAL_NETWORK_LOSS_PATH + '*')[-1]

    # eu.plot_results(results_dict)
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
