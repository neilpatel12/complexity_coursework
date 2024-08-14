#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 23:08:58 2022

@author: neilpatel
"""
#%%
import Oslo_njit as OM
import numpy as np
import time
#%%


L_list = np.array([4,8,16,32,64,128,256,512,1024,2048], dtype = int)
M = 10 #no. realisations

tmax_list = np.array([131072*2, 131072*2, 131072*2, 131072*2, 131072*2, 131072*2, 131072*2, 524288*2, 2097152*2, 16777216], dtype = int)
#%%
    
for i in range(len(L_list)):
    trialL_tcs = []
    trialL_timeit = []
    for j in range(M):
        start = time.perf_counter()
        trialL = OM.Oslo(L_list[i])
        trialL.iteration_loop(tmax_list[i])
        trialL_data = np.column_stack((trialL.t_list, trialL.height_list, trialL.s_list))
        trialL_tcs.append(trialL.tc)
        elapsed = time.perf_counter() - start
        trialL_timeit.append(elapsed)
        print(f'done in {elapsed:.02f}s') 
        np.savetxt(f'data2/trial{L_list[i]}_{j+1}.txt',trialL_data)
        
        
    np.savetxt(f'data2/trial{L_list[i]}_tcs.txt',trialL_tcs)  
    np.savetxt(f'data2/trial{L_list[i]}_timeit.txt',trialL_timeit) 
       
       
#%%
# should have data sets named for example trial4_1.txt for first realisation of system with size L. and dataset named Trial4_tcs
