# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 00:12:17 2017

@author: SrivatsanPC
"""
import pickle
import numpy as np
from scipy.special import comb
from os import chdir, getcwd
import holdem_calc as hc
import holdem_functions as hf
import itertools

chdir("E:\\CS281AdvancedML\\cs281-final-project\Programs\odds_code")
symbols = ['2','3','4','5','6','7','8','9','T','J','Q','K','A']
suits = ['h','s','c','d']
poss_cards = []
for i in symbols:
    for j in suits:
        poss_cards.append(hf.Card(i+j))

assert(len(poss_cards) == 52)

def combination_util(poss_cards, n, r,index, data, i, tot_out):
    if index == r:
        tot_out.append(data)
        return
            
    if i >= n:        
        return
    
    data[index] = poss_cards[i];
    combination_util(poss_cards, n, r, index+1, data, i+1,tot_out);
 
    combination_util(poss_cards, n, r, index, data, i+1,tot_out);
    
def generate_combinations(poss_cards, r):
    tot_out = []
    for combi in itertools.combinations(poss_cards,r):
        tot_out.append(combi)
    return tot_out

def gen_odds():
    req = [5,6]    
    for r in req:
        final_out = {}
        combos = generate_combinations(poss_cards, r) 
        i = 0
        for combo in combos:       
            board = None
            if r > 2:
                board = list(combo[2:])
            out = hc.run((tuple(combo[0:2]),),int(1e6),False,board,None,False)
            final_out = {**final_out, **out}
            i+=1
            #import pdb;pdb.set_trace()
            if i % (len(combos)//25) == 0 and i > 0:
                print("Req: " + str(r) + "- " + str(i) + " Iterations Over")
        
        pickle.dump(final_out, open("hand_eval_" + str(r) +".p", 'wb'))
        print("Set of ", r, " cards over and pickled")       
            
            
        
gen_odds()    
