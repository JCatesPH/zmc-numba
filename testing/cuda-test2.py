import math
import time

import numpy as np
import numba
from numba import cuda

import cupy as cp
from cupy import linalg

@numba.cuda.jit
def kernelfoo(x, y, N):
    tid = cuda.threadIdx.x
    blkid = cuda.blockIdx.x
    blkdim = cuda.blockDim.x

    i = tid + blkid * blkdim

    if (i >= N):
        return

    x[i] = devicefoo(i, y, N)

@numba.cuda.jit(device=True)
def devicefoo(z, y, N):
    a = z ** 2
    return a


@numba.cuda.jit
def dynamic(a, y, N):
    tid = cuda.threadIdx.x
    blkid = cuda.blockIdx.x
    blkdim = cuda.blockDim.x

    i = tid + blkid * blkdim

    if (i >= N):
        return

    y[i] = y[i] + a


N = 5
#x = cp.array([[2,0,0],[0,2,0],[0,0,2]])
x = cp.ones(N)
y = cp.zeros(N)

# Set the number of threads in a block
threadsperblock = 32

# Calculate the number of thread blocks in the grid
blockspergrid = (x.size + (threadsperblock - 1)) // threadsperblock

# Now start the kernel
tic = time.time()
kernelfoo[blockspergrid, threadsperblock](x, y, N)
toc = time.time()

print('\nx = ', x)
print('\ny = ', y)
print('\n in time ', toc-tic)



