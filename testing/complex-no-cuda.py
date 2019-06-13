#!/home/jmcates/miniconda3/envs/zmcint/bin/python
# coding: utf-8

"""
This script tests using zmcintegral on the real and imaginary parts of a complex function.
"""

# The import statements
import math
import numpy as np

def f(x):
	return complex(x[0],x[1])

def f_real(y):
	val = f(y)
	return val.real
	
def f_imag(y):
	val = f(y)
	return val.imag

xin = np.empty(2)
print('\n========================================================')
print('Function: f(x,y) = x + iy')
print('\n========================================================')
for i in range(0,10,1):
	for j in range(0,10,1):
		xin[0] = i
		xin[1] = j
		print('x = ', i, 'y = ', j, f(xin))
print('\n========================================================')

xin = [2,3]
realval = f_real(xin)
print('Real value function test: f_real(2,3) = ', realval)

imagval = f_imag(xin)
print('Real value function test: f_imag(2,3) = ', imagval)

print('Testing complete.')