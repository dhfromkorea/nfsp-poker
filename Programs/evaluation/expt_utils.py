# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 22:08:31 2017

@author: SrivatsanPC
"""

'''Win rates measured in average BB/game. mbb is a standard unit of measurement in Poker literature.
Say a game with SB = 5$ and BB = 10$. Winning 1$ in a game  would amount to a BB of 0.10.
'''

# Compares your AI agent with a random agent and plots wins.
from game import simulator
from players import *
from game.config import BLINDS
import matplotlib.pyplot as plt
from random import randint
import numpy as np


def moving_avg(x, pid, window):
    return [np.mean(x[k:k + window, pid]) for k in range(len(x) - window)]


big_blind = BLINDS[1]


def conduct_games(p1_strategy,
                  p2_strategy,
                  memory_rl_config={},
                  memory_sl_config={},
                  learn_start=128,
                  num_games=1e4,
                  num_simulations=1,
                  ret_player_ids=[0, 1],
                  mov_avg_window=5,
                  log_freq=100,
                  cuda=False,
                  verbose=False):
    # TODO:
    # 1. save_frequency -> output a pickle file which holds game results
    game_sim = simulator.Simulator(p1_strategy=p1_strategy,
                                   p2_strategy=p2_strategy,
                                   learn_start=learn_start,
                                   cuda=cuda,
                                   memory_rl_config=memory_rl_config,
                                   memory_sl_config=memory_sl_config,
                                   verbose=verbose,
                                   log_freq=log_freq)
    results = game_sim.start(num_games, return_results=True)
    # results = amount of money won/lost for each player
    # results = {player_id: +-reward}
    final_res = {}
    #print(results)
    for player_id in ret_player_ids:
        player_res = []
        for game in results.keys():
            player_res.append(np.array(list(results[game])) / big_blind)
        if mov_avg_window > 0:
            final_res[player_id] = moving_avg(np.array(player_res), player_id, mov_avg_window)
        else:
            final_res[player_id] = player_res
    return final_res


# Keys are expected to be experiment descriptions.
# Values are expected to be results.
def plot_results(results_dict, show=False, save=False, p_id=0, plot_id=randint(1, 1e8)):
    title_font = {'fontname': 'Arial', 'size': '16', 'color': 'black', 'weight': 'normal',
                  'verticalalignment': 'bottom'}  # Bottom vertical alignment for more space
    axis_font = {'fontname': 'Arial', 'size': '14'}
    for res in results_dict.keys():
        plt.plot(range(len(results_dict[res][p_id])), results_dict[res][p_id], label=res)

    plt.ylabel("Milli Big Blinds/game", **axis_font)
    plt.xlabel("Game Number", **axis_font)
    plt.title("WinRates in different experiments", **title_font)
    if show:
        plt.show()
    else:
        plt.savefig("WinRates_{}".format(plot_id), ppi=300, bbox_inches='tight')

    print("Process done and plots saved")
