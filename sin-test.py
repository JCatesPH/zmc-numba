import math
from numba import cuda
from ZMCintegral import ZMCintegral

# user defined function
@cuda.jit(device=True)
def my_func(x):
    return math.sin(x[0]+x[1]+x[2]+x[3])

MC = ZMCintegral.MCintegral(my_func,[[0,1],[0,2],[0,5],[0,0.6]])

MC.depth = 2
MC.sigma_multiplication = 5
MC.num_trials = 5

# obtaining the result
result = MC.evaluate()

# print the formatted result
print('result = %s    std = %s' % (result[0], result[1]))

