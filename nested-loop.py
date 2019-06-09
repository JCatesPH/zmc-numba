import math
import time

import numba
import numpy as np
from numba import cuda

import ZMCIntegral

@numba.cuda.jit(device=True)
def testfoo(x):
    N = 3
    result = 0
    arr = numba.cuda.shared.array((2,N), dtype=numba.types.f8)
    arr[0,0] = 1.5
    arr[0,1] = 2.5
    arr[0,2] = 3.5
    arr[1,0] = 4.5
    arr[1,1] = 5.5
    arr[1,2] = 6.5
    for i in range(0,N):
        for j in range(0,N):
            term = x[0] * arr[0,i] + x[1] * arr[1,j]
            result = result + term
    
    return result



print('================================================================')
MC = ZMCIntegral.MCintegral(testfoo,[[1,10],[1,10]])
# Setting the zmcintegral parameters
MC.depth = 2
MC.sigma_multiplication = 1E6
MC.num_trials = 5
start = time.time()
result = MC.evaluate()
print('Result: ', result[0], ' with error: ', result[1])
print('================================================================')
end = time.time()
print('Computed in ', end-start, ' seconds.')
print('================================================================')

@numba.cuda.jit(device=True)
def testfoo1(x):
    N = 3
    result = 0
    arr = numba.cuda.shared.array((2,N), dtype=numba.types.f8)
    arr[0,0] = 1.5
    arr[0,1] = 2.5
    arr[0,2] = 3.5
    arr[1,0] = 4.5
    arr[1,1] = 5.5
    arr[1,2] = 6.5
    numba.cuda.to_device(arr)
    for i in range(0,N):
        for j in range(0,N):
            term = x[0] * arr[0,i] + x[1] * arr[1,j]
            result = result + term
    
    return result


    
print('================================================================')
MC = ZMCIntegral.MCintegral(testfoo1,[[1,10],[1,10]])
# Setting the zmcintegral parameters
MC.depth = 2
MC.sigma_multiplication = 1E6
MC.num_trials = 5
start = time.time()
result = MC.evaluate()
print('Result: ', result[0], ' with error: ', result[1])
print('================================================================')
end = time.time()
print('Computed in ', end-start, ' seconds.')
print('================================================================')