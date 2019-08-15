import math
import time
from numba import cuda
from ZMCintegral import ZMCintegral

# user defined function
@cuda.jit(device=True)
def my_func(x):
    value = 0
    for i in range(5):
        value = value + math.cos(math.log(x[i]) / x[i]) / x[i]
    return value

MC = ZMCintegral.MCintegral(my_func,[[-1,1],[-1,1],[-1,1],[-1,1],[-1,1]])

MC.depth = 2
MC.sigma_multiplication = 20
MC.num_trials = 5


start = time.time()
# obtaining the result
result = MC.evaluate()

end = time.time()

# print the formatted result
print('result = %s    std = %s' % (result[0], result[1]))


print('Time to calculate: %5.4f s' % (end-start))
