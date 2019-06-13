#!/home/jalenc/.conda/envs/ZMCIntegral/bin/python
# /home/jmcates/miniconda3/envs/zmcint/bin/python
# coding: utf-8

"""
This script tests using zmcintegral on the real and imaginary parts of a complex function.
"""

# The import statements
import math
from numba import cuda
import ZMCIntegral
import numpy as np

def f(x1, x2):
    return complex(x1,x2)


@cuda.jit(device=True)
def f_real(y):
    val = f(y[0], y[1]).real
    return val

@cuda.jit(device=True)
def f_imag(y):
    val = f(y[0], y[1]).imag
    return val




MC_real = ZMCIntegral.MCintegral(f_real,[[0,1],[0,1]])

# Setting the zmcintegral parameters
MC_real.depth = 2
MC_real.sigma_multiplication = 4
MC_real.num_trials = 3
MC_real.available_GPU=[0]

MC_imag = ZMCIntegral.MCintegral(f_imag,[[0,1],[0,1]])

# Setting the zmcintegral parameters
MC_imag.depth = 2
MC_imag.sigma_multiplication = 4
MC_imag.num_trials = 3
MC_imag.available_GPU=[0]


print('\n========================================================')
print('\ndepth = ', MC_real.depth)
print('sigma_multiplication = ', MC_real.sigma_multiplication)
print('num_trials = ', MC_real.num_trials)
print('available_GPU = ', MC_real.available_GPU)
print('\n========================================================')

# Evaluating the integral
real_result = MC_real.evaluate()
imag_result = MC_imag.evaluate()

print('\nIntegration is complete!')
print('\n========================================================')
print('REAL PART:')
print('Result: ', real_result[0])
print('std.  : ', real_result[1])
print('IMAG PART:')
print('Result: ', imag_result[0])
print('std.  : ', imag_result[1])
print('\n========================================================')
