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


if __name__ == '__main__':
    cuda = False
    verbose = False
    log_freq = 1000
    num_games = 1000
    mov_avg_window = 100
    learn_start = 2 ** 8
    batch_size = 2 ** 5
    buffer_size = 2 ** 10
    num_partitions = 2 ** 4
    total_steps = 10 ** 9
    eta_p1 = .1
    eta_p2 = .1
    skip_simulation = False
    eps = .05
    gamma = 1.
    learning_rate = 1e-3
    target_Q_update_freq = 200

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
        results_dict['NFSP vs NFSP'] = eu.conduct_games('NFSP', 'NFSP',
                                                        learn_start=learn_start,
                                                        num_games=num_games,
                                                        mov_avg_window=mov_avg_window,
                                                        log_freq=log_freq,
                                                        eta_p1=eta_p1,
                                                        eta_p2=eta_p2,
                                                        eps=eps,
                                                        gamma=gamma,
                                                        learning_rate_rl=learning_rate,
                                                        learning_rate_sl=learning_rate,
                                                        target_Q_update_freq=target_Q_update_freq,
                                                        memory_rl_config=memory_rl_config,
                                                        memory_sl_config=memory_sl_config,
                                                        cuda=cuda,
                                                        verbose=verbose,
                                                        learning_freq=1,
                                                        use_batch_norm=True,
                                                        optimizer='adam',
                                                        grad_clip=None,
                                                        experiment_id=0
                                                        )

        # pick the latest created file == results just created from the simulation above
        # game_score_history_paths = g.glob(GAME_SCORE_HISTORY_PATH + '*')[-1]
        # play_history_paths = g.glob(PLAY_HISTORY_PATH + '*')[-1]
        # neural_network_history_paths = g.glob(NEURAL_NETWORK_HISTORY_PATH + '*')[-1]
        # neural_network_loss_paths = g.glob(NEURAL_NETWORK_LOSS_PATH + '*')[-1]
        #
        # # eu.plot_results(results_dict)
        # print('game history')
        # print(load_results(game_score_history_paths))
        # print('play history')
        # for k, v in load_results(play_history_paths).items():
        #     print(k)
        #     print(v)
        #     print('')
        # print('neural network history')
        # print(load_results(neural_network_history_paths))
        # print(load_results(neural_network_loss_paths))
