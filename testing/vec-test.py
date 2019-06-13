import math
import numpy as np
# from numba import cuda
import ZMCIntegral
import numba

@numba.cuda.jit(device=True)
def testfoo(x):
	N = 3
	n = 0
	f = numba.cuda.shared.array(N,dtype=numba.types.float64)
	for i in range(-(N-1)/2,(N-1)/2):
		f[n] = math.sin(x[0]+x[1])
		n = n + 1
	return f[0] + f[N-1]


x = np.linspace(0,10,1)
print('Creating ZMCintegral object')
MC = ZMCIntegral.MCintegral(testfoo, [[0,2],[0,1]])
print('====================================================')
print('Evaluating..')
results = MC.evaluate()

print('result = ', results[0])
print('error = ', results[1])

print('Complete!')
