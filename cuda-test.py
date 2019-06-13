import math
import time

import numpy as np
import numba
from numba import cuda

import cupy as cp

#@numba.cuda.jit(device=True)
def nestedfoo(x):
    y = cp.eye(3, dtype=cupy.float)
    return y


@numba.cuda.jit
def nestedfoo2(x, N):
    tid = cuda.threadIdx.x
    blkid = cuda.blockIdx.x
    blkdim = cuda.blockDim.x

    i = tid + blkid * blkdim

    if (i >= N):
        return

    x[i] = math.sin(float(i))


N = 1000000
# Create the data array - usually initialized some other way
data = np.ones(N)

# Set the number of threads in a block
threadsperblock = 32

# Calculate the number of thread blocks in the grid
blockspergrid = (data.size + (threadsperblock - 1)) // threadsperblock

# Now start the kernel
nestedfoo2[blockspergrid, threadsperblock](data, N)

# Print the result
tic = time.time()
npsum = data.sum()
toc = time.time()
print('npsum gives ', npsum)
print(' in time ', toc-tic)


with cp.cuda.Device(0):
    cpdat = cp.asarray(data)
tic = time.time()
with cp.cuda.Device(0):
    cpsum = cp.sum(cpdat)
toc = time.time()
print('cpsum gives ', cpsum)
print(' in time ', toc-tic)


