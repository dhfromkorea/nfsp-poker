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

from pycrayon import CrayonClient
import time

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
    parser.add_argument('-s1', default='NFSP', dest='strategy1')
    parser.add_argument('-s2', default='random', dest='strategy2')
    parser.add_argument('-c', '--cuda', action='store_true', dest='cuda')
    parser.set_defaults(cuda=False)
    parser.add_argument('-v', '--verbose', action='store_true', dest='verbose')
    parser.set_defaults(verbose=False)
    parser.add_argument('--mov_avg_window', default=100, type=int, dest='mov_avg_window',
                        help='moving average for game results')

    parser.add_argument('-ss', '--skip_simulation', action='store_true',
                        dest='skip_simulation', help='show only the latest results without simulation')
    parser.set_defaults(skip_simulation=False)
    parser.add_argument('-en', '--experiment_name', default='abc', type=str, help="name to be displayed in tensorboard", dest="experiment_name")
    # game/experiment
    parser.add_argument('-ng', '--num_games', default=10000, type=int, dest='num_games',
                        help='number of games to simulate')
    parser.add_argument('-lf', '--log_freq', default=1000, type=int, dest='log_freq', help='log game results frequency per game')
    parser.add_argument('-ls', '--learn_start', default=2 ** 7, type=int, dest='learn_start',
                        help='starting point for training networks')
    # experience replay
    parser.add_argument('-bs_rl', '--batch_size_rl', default=2 ** 5, type=int, dest='batch_size_rl',
                        help='batch size of Memory RL')
    parser.add_argument('-bfs_rl', '--buffer_size_rl', default=2 ** 17, type=int, dest='buffer_size_rl',
                        help='buffer size of Memory RL')
    parser.add_argument('-bs_sl', '--batch_size_sl', default=2 ** 5, type=int, dest='batch_size_sl',
                        help='batch size of Memory SL')
    parser.add_argument('-bfs_sl', '--buffer_size_sl', default=2 ** 17, type=int, dest='buffer_size_sl',
                        help='buffer size of Memory SL')
    parser.add_argument('-np', '--num_partitions', default=2 ** 11, type=int, dest='num_partitions',
                        help='number of partitions to Memory RL')
    parser.add_argument('-ts', '--total_steps', default=10 ** 9, type=int, dest='total_steps',
                        help='total steps to Memory RL')

    # define neural network parameters here
    # default values are set close to NFSP paper
    # note there's an internal epsilon decay schedule
    parser.add_argument('-eps', '--epsilon', default=0.1, type=float, dest='eps',
                        help='eps')
    parser.add_argument('-g', '--gamma', default=0.95, type=float, dest='gamma',
                        help='gamma')
    parser.add_argument('-lr_rl', '--learning_rate_rl', default=0.001, type=float,
                        dest='learning_rate_rl', help='learning rate for memory rl')
    parser.add_argument('-lr_sl', '--learning_rate_sl', default=0.0001, type=float,
                        dest='learning_rate_sl', help='learning rate for memory sl')
    parser.add_argument('-tf', '--target_Q_update_freq', default=100, type=int,
                        dest='target_Q_update_freq', help='update target Q every X number of episodes')
    parser.add_argument('-ep1', '--eta_p1', default=0.1, type=float, dest='eta_p1',
                        help='eta for player 1')
    parser.add_argument('-ep2', '--eta_p2', default=0.1, type=float, dest='eta_p2',
                        help='eta for player 2')
    # flipped logic but let's keep this for now
    parser.add_argument('-bn', '--use_batch_norm', action='store_true', dest='use_batch_norm')
    parser.set_defaults(use_batch_norm=False)
    parser.add_argument('-opt', '--optimizer', default='adam', type=str, help="optimizer to use",
                        dest="optimizer")
    parser.add_argument('-gc', '--gradient_clip', default=None, type=float, help="max l2 gradient norm", dest="grad_clip")
    # default is eery episode
    parser.add_argument('-lrnf', '--learning_frequency', default=1, type=int, dest='learning_freq',
                        help='performing backprop every how many episodes')
    parser.add_argument('-tbhn', '--tensorboard_hostname', default='http://192.168.99.100',
                        type=str, dest='tb_hostname', help='hostname for tensorboard')
    parser.add_argument('-tbp', '--tensorboard_port', default='8889', type=str, dest='tb_port',
                        help='port for tensorboard')
    return parser

def setup_tensorboard(exp_id, cur_t, hostname, port):
    exp_filename = '{}_{}'.format(cur_t, exp_id)
    tb = CrayonClient(hostname=hostname, port=port)
    try:
        tb_experiment = tb.create_experiment(exp_filename)
    except:
        # flush the data anew
        tb.remove_experiment(exp_filename)
        tb_experiment = tb.create_experiment(exp_filename)
    return tb_experiment, tb

def remove_all_experiments(hostname, port):
    '''
    DANGER: don't use this, unless you're sure
    '''
    tb = CrayonClient(hostname=hostname, port=port)
    tb.remove_all_experiments()


if __name__ == '__main__':
    '''
    If you want to play around with the hyperparameters, you can do:

    (e.g.) python perf_eval_experiments.py -v -c -ng 100 -ls 256 -bs_rl 32 -bfs_rl 128 -bs_sl 32 -bfs_sl 256 -np 16

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

    batch_size_rl = args.batch_size_rl
    buffer_size_rl = args.buffer_size_rl
    batch_size_sl = args.batch_size_sl
    buffer_size_sl = args.buffer_size_sl

    num_partitions = args.num_partitions
    total_steps = args.total_steps
    eta_p1 = args.eta_p1
    eta_p2 = args.eta_p2
    skip_simulation = args.skip_simulation
    eps = args.eps
    gamma = args.gamma
    learning_rate_rl = args.learning_rate_rl
    learning_rate_sl = args.learning_rate_sl
    target_Q_update_freq = args.target_Q_update_freq
    use_batch_norm = args.use_batch_norm
    optimizer = args.optimizer
    grad_clip = args.grad_clip
    learning_freq = args.learning_freq
    strategy1 = args.strategy1
    strategy2 = args.strategy2
    tb_hostname = args.tb_hostname
    tb_port = args.tb_port

    experiment_name = ''
    print('running tests with the following setup')
    for k, v in vars(args).items():
        print('{}: {}'.format(k, v))
        experiment_name += '{}:{}_'.format(k, v)
    experiment_id = '{}vs{}_{}'.format(strategy1, strategy2, hash(experiment_name)).lower()
    cur_t = time.strftime('%y%m%d_%H%M%S', time.gmtime())
    with open('data/experiment_log.txt', 'a') as f:
        f.write('{}\n{}\n'.format(experiment_id, experiment_name, cur_t))
    tb_experiment, _ = setup_tensorboard(experiment_id, cur_t, tb_hostname, tb_port)

    results_dict = {}
    if not skip_simulation:
        # sometimes we want to skip simulation and view only the latest simulation results
        memory_rl_config = {
            # 'size': 2 ** 17,
            'size': buffer_size_rl,
            # 'partition_num': 2 ** 11,
            'partition_num': num_partitions,
            # 'total_step': 10 ** 9,
            'total_step': total_steps,
            # 'batch_size': 2 ** 5
            'batch_size': batch_size_rl
        }
        memory_sl_config = {
            'size': buffer_size_sl,
            'batch_size': batch_size_sl
        }
        results_dict[strategy1 + 'vs' + strategy2] = eu.conduct_games(strategy1, strategy2,
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
                                                          learning_rate_rl=learning_rate_rl,
                                                          learning_rate_sl=learning_rate_sl,
                                                          optimizer=optimizer,
                                                          grad_clip=grad_clip,
                                                          learning_freq=learning_freq,
                                                          # default 100 episodes
                                                          target_Q_update_freq=target_Q_update_freq,
                                                          memory_rl_config=memory_rl_config,
                                                          memory_sl_config=memory_sl_config,
                                                          # default false
                                                          cuda=cuda,
                                                          verbose=verbose,
                                                          tensorboard=tb_experiment,
                                                          experiment_id=experiment_id,
                                                          use_batch_norm=use_batch_norm
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
