#!/home/jmcates/miniconda3/envs/zmcint/bin/python
# coding: utf-8

# # Proper interpreter:
# /share/apps/python_shared/3.6.5/bin/python

# # Testing example from github to see ZMCIntegral is working correctly.
# https://github.com/Letianwu/ZMCintegral
#
# It is the integration of this function:
# https://github.com/Letianwu/ZMCintegral/blob/master/examples/example01.png?raw=true


import math
from numba import cuda
import ZMCintegral
import time
import numpy as np


# user defined function
@cuda.jit(device=True)
def my_func(x):
    return math.sin(x[0]+x[1]+x[2]+x[3])

MC = ZMCintegral.MCintegral(my_func,[[0,1],[0,2],[0,5],[0,0.6]])

# Setting the initial values
MC.depth = 2
MC.sigma_multiplication = 5
MC.num_trials = 5
MC.available_GPU=[0]

## Testing the speed with different sigma multiplications
# Inititializing arrays
times = np.zeros(5)
results = np.zeros(10)
timemin = 100000

j = 0
k = 0
# Running the loop
for i in range(4,21,4):
    MC.sigma_multiplication = i
    t0 = time.time()
    result = MC.evaluate()
    t1 = time.time()
    times[j] = t1-t0
    results[k] = result[0]
    results[k+1] = result[1]
    if (times[j] < timemin):
        timemin = times[j]
        optimal_sigma = i
    j = j + 1
    k = k + 2

# Formatting prints
print('======================================================================')
print('sigma multiplication |     result     |      std      |       time (s)')
print('======================================================================')

j = 0
k = 0
# Printing results
for i in range(4,21,4):
    print("%19d  |  %12f  |  %11f  |  %15f" % (i, results[k], results[k+1], times[j]))
    j = j + 1
    k = k + 2

print('======================================================================')

print('\n The shortest calculation time was ', timemin, 'seconds with an optimal value of sigma multiplication: ', optimal_sigma,'\n')

MC.sigma_multiplication = optimal_sigma

## Testing the speed with different trial numbers
# Inititializing arrays
times = np.zeros(5)
results = np.zeros(10)
timemin = 100000

j = 0
k = 0
# Running the loop
for i in range(2,11,2):
    MC.num_trials = i
    t0 = time.time()
    result = MC.evaluate()
    t1 = time.time()
    times[j] = t1-t0
    results[k] = result[0]
    results[k+1] = result[1]
    if (times[j] < timemin):
        timemin = times[j]
        optimal_trials = i
    j = j + 1
    k = k + 2

# Formatting prints
print('======================================================================')
print('number of trials |     result     |      std      |       time (s)')
print('======================================================================')

j = 0
k = 0
# Printing results
for i in range(2,11,2):
    print("%19d  |  %12f  |  %11f  |  %15f" % (i, results[k], results[k+1], times[j]))
    j = j + 1
    k = k + 2
print('======================================================================')

print('\n The shortest calculation time was ', timemin, 'seconds with an optimal value of trials: ', optimal_trials,'\n')
print('======================================================================')
print('Time Optimization Results:')
print('     Optimal Time: ', timemin)
print('     Optimal Sigma Multiplication: ', optimal_sigma)
print('     Optimal Trials: ', optimal_trials)
