 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tues Feb 15 14:17:05 2022
 
@author: neilpatel
"""


#%%
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
from numba import njit


# use ext prefix if there is corresponding internal function
@njit
def ext_choose_z_th():
    #return rnd.choice(np.array([1,2]),p=np.array([0.5,0.5]))
    return rnd.randint(1,2+1)

@njit
def ext_update_supcrit_indices(L, zs, zs_th):
    supcrit_indices = []
    for i in range(L):
        if zs[i] > zs_th[i]:
            supcrit_indices.append(i)
    
    if len(supcrit_indices) > 0:
        stable = False
        
    else: 
        stable = True
    
    return (supcrit_indices, stable)

@njit
def ext_relaxation_event(zs, L, supcrit_indices_as_array, zs_th):
    reached_tc = False
    s_to_add = 0
    for j in supcrit_indices_as_array:
        if j == 0: # i = 1
            zs[0] -= 2
            zs[1] += 1
            s_to_add +=1
        elif j == (L - 1): # i = L
            zs[L-1] -= 1 # z_L
            zs[L-2] += 1 # z_(L-1)
            s_to_add +=1
            reached_tc = True
        else:
            zs[j] -= 2
            zs[j +1] += 1
            zs[j -1] += 1 
            s_to_add +=1
        zs_th[j] = ext_choose_z_th() # update threshold values of the supercritical lattice positions
    
    return(zs, zs_th, reached_tc, s_to_add)
    
class Oslo: # each run gives the avalanche size
   
   
    #INITIALISATION
    def __init__(self, L, _print_ = False):
        self._print_ = _print_
        self.L = L
        self.zs = np.array([0 for x in range(L)])
        self.zs_th = np.array([self.choose_z_th() for x in range(L)]) # new thresholds every relaxation event
        self.supcrit_indices = self.update_supcrit_indices()
        #self.update_supcrit_indices()
        self.s = 0
        self.s_list = []
        self.t = 0
        self.t_list = []
        self.height = None
        self.height_list = []
        self.tc_list = []
        self.grains_added = 0
        if len(self.supcrit_indices) > 0:
            self.stable = False
        else:
            self.stable = True
           
        if self._print_:
            print(f'intial zs = {self.zs}')
            print(f'intial zs_th = {self.zs_th}')
       
    #DRIVE
    def drive(self):
        if self._print_:
            print('')
            print('DRIVE')
        self.zs[0] += 1
        self.t += 1      # time, i.e. num of grains added in total
        self.t_list.append(self.t)
        if self._print_:
            print(f'Placed grain {self.t}')
            print(self.zs)
       
    #RELAXATION
    def choose_z_th(self):
        return rnd.choice([1,2], p=[0.5,0.5])
   
    def update_supcrit_indices(self):
        self.supcrit_indices, self.stable = ext_update_supcrit_indices(self.L, self.zs, self.zs_th)

        if self._print_:
            print(f'supcrit_indices are {self.supcrit_indices}')

        return self.supcrit_indices # any way you can not 'return' at end?
   
    def relaxation_event(self):
        self.zs, self.zs_th, reached_tc, s_to_add = ext_relaxation_event(self.zs, self.L, np.array(self.supcrit_indices), self.zs_th)
 
        self.s += s_to_add
        if reached_tc:
            self.tc_list.append(self.t)
            
        if self._print_:
            print(f'relaxation step {self.s} gives {self.zs}')     

        # update supcrit
        self.update_supcrit_indices()  
    
# =============================================================================
#     def relaxation_event(self):
#         self.s +=1
#         self.zs, self.zs_th, reached_tc = ext_relaxation_event(self.zs, self.L, np.array(self.supcrit_indices), self.zs_th)
#  
#         if reached_tc:
#             self.tc_list.append(self.t)
#             
#         if self._print_:
#             print(f'relaxation step {self.s} gives {self.zs}')     
# 
#         # update supcrit
#         self.update_supcrit_indices()   
# =============================================================================
           
    def relaxation_process(self):
        if self._print_:
            print('RELAXATION')
        self.s = 0 #reset avalanche size for next relaxation process
        self.update_supcrit_indices()
        #print(f'stable? {self.stable}')
        while not self.stable:
            self.relaxation_event()
            #print(f'relaxation step {self.s} gives {self.zs}')
              
        self.s_list.append(self.s)
        if self._print_:
            print(f'the avalanche size is {self.s} at time {self.t}')
        
    #ITERATION
    def iterate(self):
        self.drive()
       
        self.relaxation_process()
        self.measure_height()
       
        #TESTS
        #https://stackoverflow.com/questions/20229822/check-if-all-values-in-list-are-greater-than-a-certain-number
        assert all(z >= 0 for z in self.zs), 'zs should not be negative'
        assert all(z <= 2 for z in self.zs), 'zs should not be greater than 2 after relaxation'
        # or greater than 3 before relaxation but not sure if want to do this test
        assert self.height <= 2*self.L, 'height at site i should be less than 2L'
        
       
    def iteration_loop(self, no_trials):
        for i in range(no_trials):
            self.iterate()
        self.tc = self.tc_list[0]
        print(f'The cut-off time for system of size {self.L} is {self.tc}')
       
    #HEIGHT
    def measure_height(self):
        self.height = sum(self.zs)
        self.height_list.append(self.height)
        if self._print_:
            print(f'the height at posn i = 1 is {self.height} at time {self.t}')
        
    def get_recurrent_state_heights(self):
        tc_approx = self.L*self.L
        self.recurrent_state_heights = [self.height_list[i] for i in range(self.t_list) if self.t_list[i] > tc_approx]
        
        
    def time_average_heights(self):
        self.get_recurrent_state_heights()
        self.mean_recur_height = np.mean(self.recurrent_state_heights)
        self.recur_height_std = np.std(self.recurrent_state_heights)
        
    def height_prob(self, h):
        self.get_recurrent_state_heights()
        num_configs = (self.recurrent_state_heights).count(h)
        prob_h = num_configs/len(self.recurrent_state_heights)
        print(f'the probability of measuring height {h} is {prob_h} for system size L = {self.L}')
        return prob_h
    
    def normalisation_check(self):
        total_prob = 0
        for h in set(self.recurrent_state_heights):
            total_prob += self.height_prob(h)
        print(f'the total height probability is {total_prob}, it should be 1')
            
    
    #PLOTS
    def s_time_series(self):
        plt.plot(self.t_list, self.s_list)
        plt.title(f'Avalanche time series, L = {self.L}')
        plt.ylabel('s')
        plt.xlabel('t')
        plt.show()
       
    def height_time_series(self):
        plt.plot(self.t_list, self.height_list)
        plt.title(f'Height time series, L = {self.L}')
        plt.ylabel('Height at i = 1')
        plt.xlabel('t')
        plt.show()
   
    def smooth_height_time_series(self, M):
        self.smooth_heights(M)
        plt.plot(self.t_list_crop, self.smooth_height_list)
        plt.title(f'Smoothed height time series, L = {self.L}, M = {self.M}')
        plt.ylabel('Height at i = 1')
        plt.xlabel('t')
        plt.show()
       
    def height_time_scaling(self, M):
        self.smooth_heights(M)
        self.scaled_t_list_crop = np.array(self.t_list_crop)/self.L
        self.scaled_smooth_height_list = np.array(self.smooth_height_list)/(self.L * self.L)
        plt.plot(self.scaled_t_list_crop, self.scaled_smooth_height_list)
        plt.title(f'Scaled smoothed height time series, L = {self.L}, M = {self.M}')
        plt.ylabel('Height at i = 1')
        plt.xlabel('t')
        plt.show()
    
