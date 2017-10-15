# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 22:52:03 2017

@author: SrivatsanPC
"""

import random


def evaluate(player):
    value = player.get_value()


def calc_bet(player):
    max_bet = player.stack - player.to_play
    min_bet = player.to_play

    if max_bet < min_bet:
        min_bet = max_bet
    print('max bet ' + str(max_bet))
    print('min be  ' + str(min_bet))

    if max_bet < 0:
        max_bet = player.stack
        bet_amount = random.randrange(min_bet, max_bet + 1, 5)
        return bet_amount


class Strategy():
    def __init__(self, player):
        self.tight = 0
        self.aggression = 0
        self.cool = 0
        self.player = player
        self.name = str(self.__class__.__name__)

        # You can define playing style

    @property
    def play_style(self):
        pass

    # Gives your playing decisions. Important function needed for any strategy.
    def decide_play(self, player, pot):
        pass


class Random(Strategy):
    def decide_play(self, player, pot):
        choice = random.randint(0, 3)

        # Choice 0 - Fold,
        # Choice 1 - Call or randomly generate a bet
        # Choice 2 - Check_Call or all-in
        if choice == 0:
            player.fold(pot)
        elif choice == 1:
            if player.stack <= player.to_play:
                player.check_call(pot)
            else:
                player.bet(pot, calc_bet(player))
        elif choice == 2:
            if player.stack <= player.to_play:
                player.check_call(pot)
            else:
                player.bet(pot, player.stack)
