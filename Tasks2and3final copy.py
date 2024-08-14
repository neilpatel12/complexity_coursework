#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 23:08:15 2022

@author: neilpatel
***************************
"""
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
#%%
L_list = np.array([4,8,16,32,64,128,256,512,1024,2048], dtype = int)
N = len(L_list)
M = 10 #no. realisations
tmax_list = np.array([131072*2, 131072*2, 131072*2, 131072*2, 131072*2, 131072*2, 131072*2, 524288*2, 2097152*2, 16777216], dtype = int)
color_list = np.array(['red', 'orange', 'yellow', 'lightgreen', 'green', 'lightblue', 'blue', 'indigo', 'purple', 'magenta'])


L_list = np.array([4,8,16,32,64,128,256,512], dtype = int)
N = len(L_list)
M = 10 #no. realisations
tmax_list = np.array([131072*2, 131072*2, 131072*2, 131072*2, 131072*2, 131072*2, 131072*2, 524288*2], dtype = int)
color_list = np.array(['red', 'orange', 'yellow', 'lightgreen', 'green', 'lightblue', 'blue', 'indigo', 'purple', 'magenta'])

def smooth_heights(data, L_index, M):
    height_array = np.zeros(len(data[L_index][0,:,1]))
    for j in range(M):
           height_array += data[L_index][j,:,1]        
    smooth_heights = height_array/M       
    return smooth_heights


data = []
tc_data_list = []


for i in range(N):
    data_for_L = []
    for j in range(M):
        start = time.perf_counter()
        data_for_LR = np.loadtxt(f'data1/trial{L_list[i]}_{j+1}.txt')
        data_for_L.append(data_for_LR)
        elapsed = time.perf_counter() - start
        print(f'done in {elapsed:.02f}s') 
    data.append(np.array(data_for_L))  
    tc_data_list.append(np.loadtxt(f'data1/trial{L_list[i]}_tcs.txt'))  
    
    
    #%%
    
print('smoothing heights')    
smooth_heights_list = []
for i in range(N):
    smooth_heights_list.append(smooth_heights(data, i, M))

print('finding avg and std of tcs')
tc_avg_list = []
tc_std_list = []
for i in range(N):
    tc_avg_list.append(np.mean(tc_data_list[i]))
    tc_std_list.append(np.std(tc_data_list[i]))
    
#%%
#Task2    
def get_recurrent_state_heights(data, L_index, R_index = 0):
    tc_approx = L_list[L_index] * L_list[L_index]
    recurrent_state_heights = [data[L_index][R_index,i,1] for i in range(len(data[L_index][R_index,:,0])) if data[L_index][R_index,i,0] > tc_approx]
    return recurrent_state_heights
    
def time_average_heights(data, L_index, R_index = 0):
    recurrent_state_heights = get_recurrent_state_heights(data, L_index, R_index)
    mean_recur_height = np.mean(recurrent_state_heights)
    recur_height_std = np.std(recurrent_state_heights)
    return (mean_recur_height, recur_height_std)

def height_prob(data, L_index, R_index, h):
    recurrent_state_heights = get_recurrent_state_heights(data, L_index)
    num_configs = (recurrent_state_heights).count(h)
    prob_h = num_configs/len(recurrent_state_heights)
    #print(f'the probability of measuring height {h} is {prob_h} for system size L = {L_list[L_index]}')
    return prob_h

def normalisation_check_heights(data, L_index, R_index):
    total_prob = 0
    recurrent_state_heights = get_recurrent_state_heights(data, L_index, R_index)
    for h in set(recurrent_state_heights):
        total_prob += height_prob(data, L_index, R_index, h)
    print(f'the total height probability is {total_prob}, it should be 1')
    
def height_prob_dist(data, L_index, R_index):
    recurrent_state_heights = get_recurrent_state_heights(data, L_index, R_index)
    prob_list = []
    for h in set(recurrent_state_heights):
        prob_list.append(height_prob(data, L_index, R_index, h))
        
    prob_list = np.array(prob_list)
    set_of_heights_asarray = np.array(list(set(recurrent_state_heights)))
    prob_dist = np.column_stack((set_of_heights_asarray,prob_list))
    return prob_dist
    
#%%task2

#L_list
mean_height_list = []
height_std_list = []
height_prob_dist_list = []

for i in tqdm(range(N)):
   mean, std = time_average_heights(data, i, 0)
   mean_height_list.append(mean)
   height_std_list.append(std)
   prob_dist = height_prob_dist(data, i, 0)
   height_prob_dist_list.append(prob_dist)
np.asarray(mean_height_list)
np.asarray(height_std_list)
np.asarray(height_prob_dist_list)
   

#%%Task 2a
plt.figure()
for i in range(N-1):
    plt.plot(data[i][0,:,0], data[i][0,:,1], label = f'L = {L_list[i]}', color = color_list[i])
    plt.show()
plt.xlabel('Time (number of grains added)')
plt.ylabel('Height at i = 1')
plt.legend()
plt.show()

#%%Task 2b

#lin-lin
plt.figure()
plt.plot(L_list, tc_avg_list)

plt.title('lin-lin Average cut-off time, $\langle t_{c} \rangle$ against L')
plt.xlabel('System size L')
plt.ylabel('Average cut-off time')
plt.show()

#%%
#log-log
plt.figure()
plt.scatter(L_list, tc_avg_list, color = 'r', marker = 'x')
if False:
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('System size L')
    plt.ylabel('Average cut-off time')
    

L_list_crop = L_list[4:]
tc_avg_list_crop = tc_avg_list[4:]
p,V = np.polyfit(np.log(L_list_crop), np.log(tc_avg_list_crop), 1, cov = True)
plt.plot(L_list,(L_list ** p[0] * np.e**p[1]), label = f'Fit L>=64: The gradient is {p[0]:.03f} ± {V[0,0]:.03f}', color = 'blue')

plt.title(r'log-log Average cut-off time, $\langle t_c \rangle $ against L')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('System size, L')
plt.ylabel(r'Average cut-off time, $\langle t_c \rangle $')
plt.legend()
print(f'The gradient is {p[0]} ± {V[0,0]}')

#%% Task 2d
plt.figure()
for i in range(N):
    plt.plot(data[i][0,:,0], smooth_heights_list[i], label = f'L = {L_list[i]}', color = color_list[i])
    plt.show()
plt.xlabel('Time (number of grains added)')
plt.ylabel(f'Height (smoothed across {M} realisations) at i = 1')
plt.legend()
plt.show()

#%%
plt.figure()
for i in range(N):
    plt.plot(data[i][0,:,0]/(L_list[i]*L_list[i]), smooth_heights_list[i], label = f'L = {L_list[i]}', color = color_list[i])
    plt.show()
plt.xlabel('Time (number of grains added)/ L^2')
plt.xlim(-0.2,2)
plt.ylabel(f'Height (smoothed across {M} realisations) at i = 1')
plt.legend()
plt.show()

#%%

plt.figure()
for i in range(N):
    plt.plot(data[i][0,:,0]/(L_list[i]*L_list[i]), smooth_heights_list[i]/L_list[i], label = f'L = {L_list[i]}', color = color_list[i])
    plt.vlines(tc_avg_list[i]/(L_list[i]*L_list[i]), ymin = 0, ymax = 2, label = f'L = {L_list[i]}', color = color_list[i])
   
    plt.show()
plt.xlabel('Time (number of grains added)/ L^2')
plt.xlim(-0.2,2)
plt.ylabel(f'(Height (smoothed across {M} realisations) at i = 1)/L')
plt.legend()
plt.show()


#%% scaled
plt.figure()
for i in range(M):
    plt.plot(data[i][0,:,0]/(L_list[i]*L_list[i]), smooth_heights_list[i]/L_list[i], label = f'L = {L_list[i]}', color = color_list[i])
    plt.vlines(tc_avg_list[i]/(L_list[i]*L_list[i]), ymin = 0, ymax = 2, label = f'L = {L_list[i]}', color = color_list[i])
   
    plt.show()
plt.xlabel('Time (number of grains added)/ L^2')
plt.ylabel(f'(Height (smoothed across {M} realisations) at i = 1)/L')
plt.legend()
plt.show()

#%% Task 2e

#find a0
plt.figure()
plt.scatter(L_list, mean_height_list)

L_list_crop = L_list[4:]
mean_height_list_crop = mean_height_list[4:]
p_0,V_0 = np.polyfit(L_list_crop, mean_height_list_crop, 1, cov = True)

plt.plot(L_list,(L_list * p_0[0] + p_0[1]), label = f'Fit L>= 64: The gradient is {p_0[0]:.03f} ± {V_0[0,0]:.03f}', color = 'blue')

plt.legend()
plt.xlabel(r'$L$')
plt.ylabel(r'$\langle h \rangle$')
plt.title('PLot of <h> against L')
np.array(mean_height_list)

a0 = p_0[0]
a0_err = V_0[0,0]

#%%
# =============================================================================
# import scipy as sp
# from scipy import optimize, signal
# y = a0 * np.array(L_list) - np.array(mean_height_list)
# 
# plt.figure()
# plt.scatter(L_list, y)
# 
# def fit_exp(L,a0a1, w1):
#     return a0a1 * L **(1-w1)
# initial_guess = [a0*1, 0.77]
# 
# L_list_crop = L_list[4:]
# y_crop = y[4:]
# 
# po,po_cov = sp.optimize.curve_fit(fit_exp, L_list_crop, y_crop, initial_guess)
# omega1 = 1- po[1]
# a1 = po[0]/a0
# 
# plt.plot(L_list,fit_exp(L_list, *po), label=f'Fit: w1 = {omega1}')
# 
# plt.legend()
# plt.xlabel('L')
# plt.ylabel('a0L - <h>')
# plt.title('a0L - <h> against L')
# 
# 
# =============================================================================
#%%
import scipy as sp
from scipy import optimize, signal

y = 1 - np.array(mean_height_list)/(a0 * np.array(L_list))

plt.figure()
plt.scatter(L_list, y)

def fit_exp(L,a1, w1):
    return a1 * L **(-w1)
initial_guess = [a0*1, 0.7]

L_list_crop = L_list[4:]
y_crop = y[4:]

po,po_cov = sp.optimize.curve_fit(fit_exp, L_list_crop, y_crop, initial_guess)
omega1 = po[1]
a1 = po[0]

plt.plot(L_list,fit_exp(L_list, *po), label=f'Fit: w1 = {omega1}')

plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.xlabel('L')
plt.ylabel(r'$1 - \frac{<h>}{a_{0}L}$')
plt.title(r'$1 - \frac{<h>}{a_{0}L}$ against $L$')
#%%
#%%

y = np.array(mean_height_list)/(a0 * np.array(L_list))  - 1

plt.figure()
plt.scatter(L_list, y)

p_1,V_1 = np.polyfit(np.log(L_list), np.log(y), 1, cov = True)
plt.plot(L_list,(L_list ** p_1[0] * np.e ** p_1[1]), label = f'Fit L>= 64: The gradient is {p_1[0]:.03f} ± {V_1[0,0]:.03f}', color = 'blue')

omega1 = -p_1[0]
a1 = p_1[1]

plt.legend()
plt.yscale('log')
plt.xscale('log')
plt.xlabel('L')
plt.ylabel('<h>/(aoL) - 1')
plt.title('<h>/(aoL) - 1 against L')

#%%Task 2f
#lin-lin
plt.figure()
plt.plot(L_list,height_std_list)
plt.xlabel('System size L')
plt.ylabel('Standard deviation of time averaged height')
plt.show()

#%%
#log-log
plt.figure()
plt.scatter(L_list, height_std_list, color = 'r', marker = 'x')
if False:
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('System size L')
    plt.ylabel('Standard deviation of height \n measured over time in the recurrent regime')

L_list_crop = L_list[4:]
height_std_list_crop = height_std_list[4:]
p_1,V_1 = np.polyfit(np.log(L_list_crop), np.log(height_std_list_crop), 1, cov = True)
plt.plot(L_list,(L_list ** p_1[0] * np.e ** p_1[1]), label = f'Fit L>= 64: The gradient is {p_1[0]:.03f} ± {V_1[0,0]:.03f}', color = 'blue')

plt.yscale('log')
plt.xscale('log')
plt.xlabel('System size L')
plt.ylabel('Standard deviation of height \n measured over time in the recurrent regime')
plt.legend()
print(f'The gradient is {p_1[0]} ± {V_1[0,0]}')

#sigma prop L^0.298

#%%Task 2g
# ***************************
plt.figure()
for i in range(N):
    prob_dist = height_prob_dist_list[i]
    heights = prob_dist[:,0]
    probs = prob_dist[:,1]
    plt.bar(heights, probs, label = f'L = {L_list[i]}', color = color_list[i])
    normalisation_check_heights(data, i, 0)

plt.xlabel(r'Height $h$')
plt.ylabel('P(h;L)')
plt.legend()
#%%
plt.figure()
for i in range(1,N,2):
    prob_dist = height_prob_dist_list[i]
    heights = prob_dist[:,0]
    probs = prob_dist[:,1]
    plt.bar((heights - mean_height_list[i])/height_std_list[i], probs * height_std_list[i], label = f'L = {L_list[i]}', alpha = 0.5, color = color_list[i])
    normalisation_check_heights(data, i, 0)
plt.xlabel('$(h - <h>)/ \sigma$')
plt.ylabel('P(h;L)')
plt.legend()

#%% ***************************

plt.figure()
for i in range(0,N):
    prob_dist = height_prob_dist_list[i]
    heights = prob_dist[:,0]
    probs = prob_dist[:,1]
    plt.bar((heights - mean_height_list[i])/height_std_list[i], probs * height_std_list[i], label = f'L = {L_list[i]}', alpha = 0.25, color = color_list[i])
    normalisation_check_heights(data, i, 0)
plt.xlabel('$(h - <h>)/ \sigma$')
plt.ylabel('P(h;L)')
plt.legend()
#%%
#theoretical gaussian

def gauss_func(x, mu, sigma):
    gauss = 1/(sigma * np.sqrt(2 * np.pi)) * np.e ** -(((x-mu)** 2)/(2 * sigma**2))
    return gauss

i = 1
prob_dist = height_prob_dist_list[i]
heights = prob_dist[:,0]
probs = prob_dist[:,1]

gauss_1 = gauss_func(heights, mean_height_list[i], height_std_list[i])
#%%

#plotting expected theoretical gaussian for different L sizes
# should really just try for largest L value
plt.figure()
for i in range(N): #for each L size there are N realisations
    prob_dist = height_prob_dist_list[i]
    heights = prob_dist[:,0]
    probs = prob_dist[:,1]
    gauss = gauss_func(heights, mean_height_list[i], height_std_list[i])
    scaled_heights = heights/(L_list[i])
    scaled_mean_heights =  mean_height_list[i]/L_list[i]
    scaled_heights_stds = height_std_list[i]/(L_list[i] ** p_1[0])
    scaled_gauss = gauss_func(scaled_heights, scaled_mean_heights, scaled_heights_stds)
    plt.plot(scaled_heights, scaled_gauss * scaled_heights_stds , label = f'L = {L_list[i]}', color = color_list[i])
    #plt.plot(heights, gauss, label = f'L = {L_list[i]}', color = color_list[i])
    print(heights, gauss)
plt.xlabel('$(h - <h>)/ \sigma$')
plt.ylabel('P(h;L)')
plt.legend()    


#%%Task3

def get_recurrent_state_s(data, L_index, R_index = 0):
    tc_approx = L_list[L_index] * L_list[L_index]
    recurrent_state_s = [data[L_index][R_index,i,2] for i in range(len(data[L_index][R_index,:,0])) if data[L_index][R_index,i,0] > tc_approx]
    return recurrent_state_s
    

def time_average_s(data, L_index, R_index = 0):
    recurrent_state_s = get_recurrent_state_s(data, L_index, R_index)
    mean_recur_s = np.mean(recurrent_state_s)
    recur_s_std = np.std(recurrent_state_s)
    return (mean_recur_s, recur_s_std)


def s_prob(data, L_index, R_index, s):
    recurrent_state_s = get_recurrent_state_s(data, L_index)
    num_avalanches = (recurrent_state_s).count(s) # num avalanches of size s
    prob_s = num_avalanches/len(recurrent_state_s)
    #print(f'the probability of measuring height {h} is {prob_h} for system size L = {L_list[L_index]}')
    return prob_s


def normalisation_check_s(data, L_index, R_index):
    total_prob = 0
    recurrent_state_s = get_recurrent_state_s(data, L_index, R_index)
    for s in set(recurrent_state_s):
        total_prob += s_prob(data, L_index, R_index, s)
        #print(total_prob)
    print(f'the total s probability is {total_prob}, it should be 1')
  

def s_prob_dist(data, L_index, R_index):
    recurrent_state_s = get_recurrent_state_s(data, L_index, R_index)
    prob_list = []
    print(len(set(recurrent_state_s)))
    for h in set(recurrent_state_s):
        prob_list.append(s_prob(data, L_index, R_index, h))
        #print(h)
    prob_list = np.array(prob_list)
    set_of_s_asarray = np.array(list(set(recurrent_state_s)))
    prob_dist = np.column_stack((set_of_s_asarray,prob_list))
    return prob_dist

#%% task3 very slow

mean_s_list = []
s_std_list = []
#s_prob_dist_list = []

for i in tqdm(range(N)):
   mean, std = time_average_s(data, i, 0)
   mean_s_list.append(mean)
   s_std_list.append(std)

np.asarray(mean_s_list)
np.asarray(s_std_list)

#%%
from logbin_2020 import logbin

#ss = get_recurrent_state_s(data, 5, 0)
#ss_bin_centres, s_frequencies_log_binned = logbin(data = ss, scale = 1.25, zeros = False)

#%%

#large N necessary for unliekly large avalanches to be recorded
#qualitative description: straight line negative gradient then bump near cut-off avalanche size

scale = 1.5
plt.figure()
for i in range(N):
    ss = get_recurrent_state_s(data, i, 0)
    ss_bin_centres, prob_log_binned_s = logbin(data = ss, scale = scale, zeros = False)
    plt.plot(ss_bin_centres, prob_log_binned_s, label = f'L = {L_list[i]}', color = color_list[i])
    #normalisation_check_s(data, i, 0)

plt.title(f'Log-binned P(s;L) against s, with scale {scale}')
plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'$\hat{s}$')
plt.ylabel(r'$\tilde{P}(\hat{s};L)$')
plt.legend()

#%%

#DATA COLLAPSE
#should be D = 2.25 and τs = 1.55.
T_s = 1.56 #±0.01
plt.figure()
for i in range(N):

    ss = get_recurrent_state_s(data, i, 0)
    ss_bin_centres, prob_log_binned_s = logbin(data = ss, scale = scale, zeros = False)
    s_scaled_prob = [(ss_bin_centres[i] ** (T_s)) * prob_log_binned_s[i] for i in range(len(ss_bin_centres))]
    plt.plot(ss_bin_centres, s_scaled_prob, label = f'L = {L_list[i]}', color = color_list[i])

plt.title(f'Log-binned s^{T_s} * P(s;L) tilda against s')
plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'$\frac{\hat{s}}{L^{D}} $')
plt.ylabel(r'$\tilde{P}(\hat{s};L)$')
plt.legend()

#%%
D = 2.12 #± 0.05
plt.figure()
for i in range(N):

    ss = get_recurrent_state_s(data, i, 0)
    ss_bin_centres, prob_log_binned_s = logbin(data = ss, scale = scale, zeros = False)
    s_scaled_prob = [(ss_bin_centres[i] ** (T_s)) * prob_log_binned_s[i] for i in range(len(ss_bin_centres))]
    ss_bin_centres_scaled = ss_bin_centres/(L_list[i] ** D)
    plt.plot(ss_bin_centres_scaled, s_scaled_prob, label = f'L = {L_list[i]}', color = color_list[i])

plt.title(f'Log-binned s^{T_s} * P(s;L) tilda against s/L^{D}')
plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'$\frac{\hat{s}}{L^{D}} $')
plt.ylabel(r'$ s^{\tau_{s}}  \tilde{P}(\hat{s};L)$')
plt.legend()

#%% Task 3b

# =============================================================================
# for i in range(N):
#     ss = np.array(get_recurrent_state_s(data, i, 0))
#     ss2 = ss**2
#     ss3 = ss**3
#     ss4 = ss**4
#     ss_mean = np.mean(ss)
#     ss2_mean = np.mean(ss2)
#     ss3_mean = np.mean(ss3)
#     ss4_mean = np.mean(ss4)
#     ss_moms = [ss_mean, ss2_mean, ss3_mean, ss4_mean]
#     plt.plot(L_list[i], ss_moms, label = f'L = {L_list[i]}', color = color_list[i])
# 
# =============================================================================
k_list = [1,2,3,4,5,6]
L_list_crop = L_list[4:]

plt.figure()
for k in tqdm(k_list):
    mom_list = []
    for i in range(N):
        ss = np.array(get_recurrent_state_s(data, i, 0))
        mom = np.mean(ss ** k)
        mom_list.append(mom)
    plt.scatter(L_list, mom_list, label = f'$<s^{k}>$', color = color_list[k])
    print('fitting')
    mom_list_crop = mom_list[4:]
    p_1,V_1 = np.polyfit(np.log(L_list_crop), np.log(mom_list_crop), 1, cov = True)
    plt.plot(L_list,(L_list ** p_1[0] * np.e ** p_1[1]), label = f'Fit for L>=64: The gradient is {p_1[0]:.03f} ± {V_1[0,0]:.03f}', color = 'blue')
    
plt.title(r'$\langle s^k \rangle $ against $L$')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('L')
plt.ylabel(r'$\langle s^k \rangle $')
plt.legend()


#%%

grad_list = []
grad_err_list = []
plt.figure()
for k in tqdm(k_list):
    mom_list = []
    for i in range(N):
        ss = np.array(get_recurrent_state_s(data, i, 0))
        mom = np.mean(ss ** k)
        mom_list.append(mom)
    #plt.scatter(L_list, mom_list, label = f'$<s^{k}>$', color = color_list[k])
    print('fitting')
    p_1,V_1 = np.polyfit(np.log(L_list), np.log(mom_list), 1, cov = True)
    #plt.plot(L_list,(L_list ** p_1[0] * np.e ** p_1[1]), label = f'Fit: The gradient is {p_1[0]:.03f} ± {V_1[0,0]:.03f}', color = 'blue')
    grad_list.append(p_1[0])
    grad_err_list.append(V_1[0,0])

plt.scatter(k_list, grad_list)
plt.errorbar(k_list, grad_list, yerr = grad_err_list)
p_2,V_2 = np.polyfit(k_list, grad_list, 1, cov = True)
plt.plot(k_list, p_2[0]*np.array(k_list) + p_2[1], label = f'Fit: y = ({p_2[0]:.03f} ± {V_2[0,0]:.03f})k + ({p_2[1]:.03f} ± {V_2[1,1]:.03f})')


plt.title(r'$D(1 + k - \tau_s)$ against $k$')
plt.xlabel(r'$k$')    
plt.ylabel(r'$D(1 + k - \tau_s)$')
plt.legend()


























