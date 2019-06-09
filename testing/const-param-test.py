#!/home/jmcates/miniconda3/envs/zmcint/bin/python
# coding: utf-8

# The import statements
import math
import time
import numpy as np
from numba import cuda
import ZMCIntegral

# Initialize the variable qx as global
qx = 0
print('qx = ', qx)

# Helper function to set qx
# @cuda.jit(device=True)
def setqx(qxi):
	global qx
	qx = qxi
	return

# Helper function to get qx
@cuda.jit(device=True)
def getqx():
    return qx


# user defined function
@cuda.jit(device=True)
def my_func(x):
    a = getqx()
    return math.sin(x[0]+x[1]+x[2]+a)

setqx(0.1)
print('qx = ', qx)

MC = ZMCIntegral.MCintegral(my_func,[[0,1],[0,2],[0,5]])

MC.depth = 2
MC.sigma_multiplication = 5
MC.num_trials = 5
MC.available_GPU=[0]

# obtaining the result
t0 = time.time()
result = MC.evaluate()
t1 = time.time()

# print the formatted result
print('result = %s    std = %s' % (result[0], result[1]))
print('This was calculated in', t1-t0, 's')
