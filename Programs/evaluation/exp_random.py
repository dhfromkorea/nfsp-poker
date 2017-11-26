# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 19:47:40 2017

@author: SrivatsanPC
"""

'''Win rates measured in average mbb/game. mbb is a standard unit of measurement in Poker literature.
Say a game with SB = 5$ and BB = 10$. Winning 1$ in a game  would amount to a BB of 0.10.
or 10 mbb/game.mbb - milli BB is 1/1000th of a BB.
'''

#Compares your AI agent with a random agent and plots wins.
from game import simulator
from players import *

def exp_random(AI_agent_type, save = False, show_plots = True):
    sim = Simulator()

