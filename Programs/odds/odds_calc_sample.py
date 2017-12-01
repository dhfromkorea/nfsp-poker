# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 18:19:58 2017

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

# chdir("E:\\CS281AdvancedML\\cs281-final-project\Programs\odds\pickledir")
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

def generate_sample(poss_cards,r):
    random.shuffle(poss_cards)
    return poss_cards[:r]
    
def gen_odds():
    req = [7]
    no_samples = int(5e5)
       
    baseline = time.time()
    for r in req:
        i=0
        final_out = {}
        while i < no_samples:
            combo = generate_sample(poss_cards,r)     
            board = None
            if r > 2:
                board = list(combo[2:])
            out = hc.run((tuple(combo[0:2]),),int(3000),False,board,None,False)
                #pdb.set_trace()
            final_out.update(out)
                #final_out = {**final_out, **out}
            i+=1
            #import pdb;pdb.set_trace()
            if no_samples <= 5:
                print(out)
                    
            if time.time() - baseline > 900 or i % int(1e6) == 0:
                baseline = time.time()
                pickle.dump(final_out, open("hand_eval_sample" + str(r) + str(time.time()) +".p", 'wb', ))
                final_out = {}
                print(i, " iterations over")
        
        print("Set of ", r, " cards over and pickled")       
            
            
        
gen_odds()    