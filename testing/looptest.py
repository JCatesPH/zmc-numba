#!/home/jalenc/.conda/envs/ZMCIntegral/bin/python
# coding: utf-8

# # Testing example from github to see ZMCIntegral is working correctly.
# https://github.com/Letianwu/ZMCintegral
#
# It is the integration of this function:
# https://github.com/Letianwu/ZMCintegral/blob/master/examples/example01.png?raw=true

import sys
sys.path.append('/home/jmcates/ZMCIntEnv/lib/python2.7/site-packages')
# print(sys.path)

from numba import cuda
import numpy as np
import time

# Inititializing arrays
times = np.empty(5)
results = np.empty(10)
timemin = 100000
j = 0

# Running the loop
for i in range(4,21,4): # Loop over: 4, 8, 12, 16, 20
    t0 = time.time()
    result =  i**2
    t1 = time.time()
    times[j] = t1-t0
    results[j] = result
    if ((t1-t0) < timemin):
        timemin = t1-t0
        optimal_sigma = i
    j = j + 1

j = 0
# Formatting prints
print('======================================================================')
print('sigma multiplication |     result     |      std      |       time (s)')
print('======================================================================')

# Printing results
for k in range(4,21,4):
	print('%19d  |  %12f  |  %12f |  %13f' % (k, results[j], results[j], times[j]))
	j = j + 1

print('======================================================================')

print('\n The shortest calculation time was ', timemin, 'seconds with an optimal value of sigma multiplication: ', optimal_sigma,'\n')


