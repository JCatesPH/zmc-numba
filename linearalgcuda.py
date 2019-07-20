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
###########################################################################
# # Inverse for Tridiagonal Matrices
###########################################################################
@numba.cuda.jit(device=True)
def myInvTZ(N, lower, diag, upper, inv):
    '''
    A CUDA device function that computes the inverse matrix for a 
        complex-valued tridiagonal matrix. From my solution on paper 
        using naive Gauss-Jordan Elimination

    This is:  A * A**-1 = I
        A : Tridiagonal, N x N matrix
        A**-1 : Inverse of A
        I : N x N Identity matrix

    Parameters 
    ----------
        N : int 
            Size of square matrix
        lower : complex array 
            Vector of size (N-1) that is the lower diagonal entries of A
        diag : complex array 
            Vector of size N that is the diagonal entries of A
        upper : complex array 
            Vector of size (N-1) that is the upper diagonal entries of A
        inv : complex array
            N x N Identity matrix

    Returns
    -------
        inv : complex matrix
            Matrix of size N x N that is the inverse of A
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



###########################################################################
# # Trace
###########################################################################
@cuda.jit(device=True)
def trace(arr, N, tr):
    '''
    A CUDA device function that computes the trace of a complex matrix.
        In other words, the sum of the diagonal entries of an array.
    Parameters 
    ----------
        arr : complex array
            The complex (N x N) array that is having its trace computed
        N : int 
            Size of square matrix
    Returns
    -------
        tr : complex 
            The trace, or sum of diagonal entries of arr
    '''
    tr[0] = 0+0j
    for i in range(0, N):
        tr[0] = tr[0] + arr[i,i]
    
    #return tr



###########################################################################
# # Conjugate-Transpose
###########################################################################
@cuda.jit(device=True)
def TZdagger(arr, N, adag):
    '''
    A CUDA device function that computes the conjugate transpose of a 
        complex matrix.

    Parameters 
    ----------
        arr : complex array
            The complex (N x N) array that is having its trace computed
        N : int 
            Size of square matrix

    Returns
    -------
        adag : complex array
            The conjugate transpose of arr
    '''

    for i in range(0, N):
        for j in range(0, N):
            tmp = complex(arr[j, i].real, -arr[j, i].imag)
            adag[i, j] = tmp

    return adag



###########################################################################
# # Matrix Multiplication
###########################################################################
@cuda.jit(device=True)
def squareMatMul(A, B, C, N):
    '''
    A CUDA device function that multiplies two square, NxN matrices.

    AB=C

    Parameters 
    ----------
        A : NxN matrix
            First matrix to be multiplied
        B : NxN matrix
            Second matrix to be multiplied
        C : NxN matrix
            Product of AB
        N : int 
            Size of square matrix

    Returns
    -------
        C : NxN matrix
            Product of AB
    '''
    for i in range(N):
        for j in range(N):
            C[i,j] = complex(0,0)

    for i in range(N):
        for j in range(N):
            for l in range(N):
                C[i,j] = A[i,l] * B[l,j] + C[i,j]
    
    return C



############S##############################################################
# # General Inverse
###########################################################################
@cuda.jit(device=True)
def myInvSZ(A, Inverse, N):
    '''
    A CUDA device function that computes the inverse for a 
        complex-valued, square matrix.

    This is:  A * A**-1 = I
        A : Square, N x N matrix
        A**-1 : Inverse of A
        I : N x N Identity matrix

    Parameters 
    ----------
        A : complex matrix
            N x N matrix having its inverse computed
        I : complex matrix
            N x N identity matrix that will have its values altered to the inverse of A
        N : int 
            Size of square matrix
        
    Returns
    -------
        Inverse : complex matrix
            Matrix of size N x N that is the inverse of A
    '''

    # # ELIMINATE LOWER TRIANGLE
    for k in range(N-1):
        #diag = A[k,k]
        
        for i in range(k+1, N):
            #ratio =  A[i,k] / A[k,k]

            for j in range(N):
                Inverse[i,j] = Inverse[i,j] - A[i,k] / A[k,k] * Inverse[k,j]
                A[i,j] = A[i,j] - A[i,k] / A[k,k] * A[k,j]

    # # ELIMINATE UPPER TRIANGLE
    for k in range(N-1, 0, -1):
        #diag = A[k,k]
        
        for i in range(k-1, -1, -1):
            #ratio = A[i,k] / A[k,k]

            for j in range(N):
                Inverse[i,j] = Inverse[i,j] - A[i,k] / A[k,k] * Inverse[k,j]
                A[i,j] = A[i,j] - A[i,k] / A[k,k] * A[k,j]

    # # REDUCE ROWS
    for i in range(N):
        #diag = A[i,i]

        for j in range(N):
            Inverse[i,j] = Inverse[i,j] / A[i,i]

    return Inverse

#%%
#####################################################################
# # Kernel Functions
#####################################################################
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
        tr[i] = trace(array, N, tr)


@numba.cuda.jit()
def tkcj(A, N, A_dag):
    tid = cuda.threadIdx.x
    blkid = cuda.blockIdx.x
    blkdim = cuda.blockDim.x

    i = tid + blkid * blkdim
    if(i < 1):
        A_dag = TZdagger(A, N, A_dag)

@numba.cuda.jit()
def gsmm(A, B, C, N):
    tid = cuda.threadIdx.x
    blkid = cuda.blockIdx.x
    blkdim = cuda.blockDim.x

    i = tid + blkid * blkdim
    if(i < 1):
        C = squareMatMul(A, B, C, N)


#%%
# # # # # # # # # # # 
# TESTING NEW TRACE #
# # # # # # # # # # # 
A = np.array(
    [[2, 4],
     [3, 5]],
     dtype=np.complex64
)

N = 2
tr = np.array([0], dtype=np.complex64)

tktr[1, 32](A, N, tr)
print('tr = ', tr[0])


#%%
'''
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

contran = np.ones((N,N), dtype=np.complex)
tkcj[1, 32](A, N, contran)
print('A = \n', A)
print('A* = \n', contran)


N = 1000

for k in range(0,10):
    top = np.random.rand(N-1)
    bot = np.random.rand(N-1)
    inn = np.random.rand(N)

    iden = np.identity(N, dtype=np.complex)

    A = np.diag(top, k=1) + np.diag(bot, k=-1) + np.diag(inn)

    tic1 = time.time()
    inv = np.linalg.inv(A)
    toc1 = time.time()

    tic2 = time.time()
    tkinvtz[1, 32](N, bot, inn, top, iden)
    toc2 = time.time()

    print(np.isclose(inv,iden))

    print('numpy time = ', toc1-tic1)
    print('my time = ', toc2-tic2)


N = 7

mat1 = np.random.randint(-50, 50, (N,N))
mat2 = np.random.randint(-50, 50, (N,N))

res = np.zeros((N,N))

gsmm[1, 32](mat1, mat2, res, N)

test = np.matmul(mat1,mat2)

print(res)

print('\nSame:', (test==res).all())
'''
#%%
