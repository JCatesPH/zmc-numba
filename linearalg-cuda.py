'''
linearalg.py declares the following linear algebra functions to be used in cuda decorated functions. 

They must all work as device functions (without CUDA kernels)
'''
#%%
import math
import time

import numba
import numpy as np
from numba import cuda


#%%
@numba.cuda.jit(device=True)
def myInvTZ(N, lower, diag, upper, inv):
    '''
    My own function computes the inverse matrix for a complex-valued tridiagonal matrix.
   
    From my solution on paper using naive Gauss-Jordan Elimination

    This is:  A * A**-1 = I
        A : Tridiagonal, N x N matrix
        A**-1 : Inverse of A
        I : N x N Identity matrix

    INPUT:
        N : (int) Size of square matrix
        lower : (complex array) Vector of size (N-1) that is the lower diagonal entries of A
        diag : (complex array) Vector of size N that is the diagonal entries of A
        upper : (complex array) Vector of size (N-1) that is the upper diagonal entries of A
        inv : N x N Identity matrix

    OUTPUT:
        inv : (complex matrix) Matrix of size N x N that is the inverse of A
    '''
    #           0    1    2
    # lower = [a21, a32]
    # diag  = [a11, a22, a33]
    # upper = [a12, a23]
    ###########################################################################
    # # Gaussian elimination of lower triangle
    ###########################################################################
    # R(i) = R(i) - a(i+1,i) / a(i,i) * R(i-1)
    for i in range(1, N):
        for j in range(0, N):   
            inv[i,j] = inv[i,j] - lower[i-1] / diag[i-1] * inv[i-1, j]
            
        diag[i] = diag[i] - lower[i-1] / diag[i-1] * upper[i-1]
        lower[i-1] = 0
    ###########################################################################
    # # Gaussian elimination of upper triangle
    ###########################################################################
    # R(i) = R(i) - a(i+1,i+2) / a(i+1,i+2) * R(i+1)
    for i in range(N-2, -1, -1):
        for j in range(0, N):
            inv[i,j] = inv[i,j] - upper[i] / diag[i+1] * inv[i+1, j]

        upper[i] = 0
    ###########################################################################
    # # Row reduction
    ###########################################################################
    # R(i) = R(i) / a(i,i)
    for i in range(0,N):
        for j in range(0,N):
            inv[i,j] = inv[i,j] / diag[i]
    ###########################################################################

    return inv

@cuda.jit(device=True)
def trace(arr, N, tr):
    tr = 0+0j
    for i in range(0, N):
        tr = tr + arr[i,i]
    
    return tr


#%%
@numba.cuda.jit()
def tkinvtz(N, bot, inn, top, iden):
    tid = cuda.threadIdx.x
    blkid = cuda.blockIdx.x
    blkdim = cuda.blockDim.x

    i = tid + blkid * blkdim
    if(i < 1):
        iden = myInvTZ(N, bot, inn, top, iden)

@numba.cuda.jit()
def tktr(array, N, tr):
    tid = cuda.threadIdx.x
    blkid = cuda.blockIdx.x
    blkdim = cuda.blockDim.x

    i = tid + blkid * blkdim
    if(i < 1):
        tr = trace(array, N, tr)


#%%

N = 5


top = np.array([2+2j, -4-4j, 8+4j, 4-2j])
bot = np.array([2+2j,  2-2j, 2+1j, 4-2j])
inn = np.array([2+2j,  2-2j, 2-2j, 8+2j, 4+1j])

iden = np.identity(N, dtype=np.complex)

A = np.diag(top, k=1) + np.diag(bot, k=-1) + np.diag(inn)
print('A = \n', A)

np.set_printoptions(precision=4, suppress=True, linewidth=90)

tkinvtz[1, 32](N, bot, inn, top, iden)
print('Ainv = \n', iden)

b = np.matmul(A, iden)
print('b = \n', b)

print('\nnumpy result: \n', np.linalg.inv(A))



#%%

tr = 12j
tktr[1, 32](A, N, tr)

print('tr = ', tr)

#%%
