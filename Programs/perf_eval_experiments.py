# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 19:32:38 2017

@author: SrivatsanPC
"""

'''Possible Experiments
1. NFSP RL agent against a totally random agent
2. NFSP RL agent against a mirror agent
3. NFSP RL agent against a lagged agent
4. NFSP RL agent against a simple DDQN(eta = 0) and the average policy agent(eta=1)
'''

import evaluation.expt_utils as eu

#Example
results_dict = {}
results_dict['Random vs Random'] = eu.conduct_games('RL', 'RL', num_games = 10)
eu.plot_results(results_dict)

