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
import pdb
import time
from copy import deepcopy
import random, math

#chdir("E:\\CS281AdvancedML\\cs281-final-project\Programs\odds_code")
symbols = ['2','3','4','5','6','7','8','9','T','J','Q','K','A']
suits = ['h','s','c','d']
poss_cards = []
for sy in symbols:
    for su in suits:
        poss_cards.append(hf.Card(sy+su))

assert(len(poss_cards) == 52)

def ncr(n,r):
    return math.factorial(n)/(math.factorial(r)*math.factorial(n-r))
                          
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
        hole_combos = generate_combinations(poss_cards, 2) 
        overall_combos = []
        #pdb.set_trace()
        if r > 2:
            for two_combo in hole_combos:
                dummy_poss_cards = deepcopy(poss_cards)
                dummy_poss_cards.remove(two_combo[0])
                dummy_poss_cards.remove(two_combo[1])                
                rem_cards = generate_combinations(dummy_poss_cards, r-2) 
                stitched_list = [two_combo + x for x in rem_cards]
                overall_combos += stitched_list
        
        random.shuffle(overall_combos)
        print("Cards generated")
        assert(len(overall_combos) == ncr(52,2) * ncr(50,r-2))
        start = time.time()
        baseline = start
        #pdb.set_trace()
        i = 0
        for combo in overall_combos:       
            board = None
            if r > 2:
                board = list(combo[2:])
            out = hc.run((tuple(combo[0:2]),),int(1e4),False,board,None,False)
            #pdb.set_trace()
            final_out.update(out)
            #final_out = {**final_out, **out}
            i+=1
            #import pdb;pdb.set_trace()
                    
            if time.time() - baseline > 3600:
                baseline = time.time()
                pickle.dump(final_out, open("hand_eval_" + str(r) + str(time.time()) +".p", 'wb', ))
                final_out = {}
                print(i, " iterations over")
        
        print("Set of ", r, " cards over and pickled")       
            
            
        
gen_odds()    
