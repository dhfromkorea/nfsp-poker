# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 17:43:30 2017

@author: SrivatsanPC
"""

import holdem_calc as hc
import holdem_functions as hf

#Example for calculating hand strength for a pair of cards without any board. For instance,
#let us calculate it for 3s 4h.
card_1 = hf.Card('3s')
card_2 = hf.Card('4h')
combo = (card_1,card_2)
print("Crossing here")
out = hc.run((tuple(combo),),int(1e5),False,None,None,False)
print(out)

#Example for calculating hand strength including the board cards.
card_1 = hf.Card('3s')
card_2 = hf.Card('4h')
combo = (card_1,card_2)
board = [hf.Card('6c'), hf.Card('7s'), hf.Card('9d')]
out = hc.run((tuple(combo),),int(1e5),False, board,None,False)
print(out)

#Example to run with opponent and board
card_e_1 = hf.Card('5c')
card_e_2 = hf.Card('Ah')
card_1 = hf.Card('3s')
card_2 = hf.Card('4h')
combo = (card_1,card_2)
combo_e = (card_e_1,card_e_2)
board = [hf.Card('Ac'), hf.Card('Tc'), hf.Card('Qd')]
out = hc.run((tuple([combo,combo_e])),int(1e5),False, board,None,False,pad_opp=False)
print(out)
