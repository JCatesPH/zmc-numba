#!/usr/bin/python
# coding: utf-8

# # Testing example from github to see ZMCIntegral is working correctly.
# https://github.com/Letianwu/ZMCintegral
#
# It is the integration of this function:
# https://github.com/Letianwu/ZMCintegral/blob/master/examples/example01.png?raw=true

import sys
sys.path.append('/home/jmcates/ZMCIntEnv/lib/python2.7/site-packages')
# print(sys.path)

import math
from numba import cuda
from ZMCintegral import ZMCintegral
import time

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
times = zeros(5)
results = zeros(10)
timemin = 100000

# Running the loop
for i in range(4,20,4):
    MC.sigma_multiplication = i
    t0 = time.time()
    result = MC.evaluate()
    t1 = time.time()
    times[i/4-1] = t1-t0
    results[i/4-1,i/4] = result
    if ((t1-t0) < timemin):
        timemin = t1-t0
        optimal_sigma = i

# Formatting prints
print('======================================================================')
print('sigma multiplication |     result     |      std      |       time (s)')
print('======================================================================')

# Printing results
for i in range(4,20,4):
    print("%19d  |  %12f  |  %11f  |  %15f" % (i, result[i/4-1], result[i/4], times[i/4-1]))

print('======================================================================')

print('\n The shortest calculation time was ', timemin, 'seconds with an optimal value of sigma multiplication: ', optimal_sigma,'\n')

MC.sigma_multiplication = optimal_sigma

## Testing the speed with different trial numbers
# Inititializing arrays
times = zeros(5)
results = zeros(10)
timemin = 100000

# Running the loop
for i in range(2,10,2):
    MC.num_trials = i
    t0 = time.time()
    result = MC.evaluate()
    t1 = time.time()
    times[i/4-1] = t1-t0
    results[i/4-1,i/4] = result
    if ((t1-t0) < timemin):
        timemin = t1-t0
        optimal_trials = i

# Formatting prints
print('======================================================================')
print('number of trials |     result     |      std      |       time (s)')
print('======================================================================')

# Printing results
for i in range(2,10,2):
    print("%19d  |  %12f  |  %11f  |  %15f" % (i, result[i/4-1], result[i/4], times[i/4-1]))

print('======================================================================')

print('\n The shortest calculation time was ', timemin, 'seconds with an optimal value of trials: ', optimal_trials,'\n')
print('======================================================================')
print('Time Optimization Results:')
print('     Optimal Time: ', timemin)
print('     Optimal Sigma Multiplication: ', optimal_sigma)
print('     Optimal Trials: ', optimal_trials)

