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

    for i in range(0, N):
        tr = tr + arr[i,i]
    
    return tr



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
    #for i in range(N):
    #    for j in range(N):
    #        C[i,j] = complex(0,0)
    #C = numba.cuda.shared.array((N,N), dtype=numba.types.complex64)

    for i in range(N):

        for j in range(N):
            tmp = 0+0j

            for l in range(N):
                tmp += A[i,l] * B[l,j]

            C[i,j] = tmp
    
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
    tmp = cuda.local.array((5,5), dtype=numba.types.c16)

    for i in range(N):
        for j in range(N):
            tmp[i,j] = A[i,j]

    # # ELIMINATE LOWER TRIANGLE
    for k in range(N-1):

        if (tmp[k,k].real != 1) or (tmp[k,k].imag != 0):
            scale = tmp[k,k]
            for j in range(N):
                Inverse[k,j] = Inverse[k,j] / scale
                tmp[k,j] = tmp[k,j] / scale
                
        for i in range(k+1, N):
            if (tmp[i,k].real != 0) and (tmp[i,k].imag != 0):
                ratio =  tmp[i,k]

                for j in range(N):
                    tmp[i,j] = tmp[i,j] - ratio * tmp[k,j]
                    Inverse[i,j] = Inverse[i,j] - ratio * Inverse[k,j]

    if (tmp[N-1,N-1].real != 1) or (tmp[N-1,N-1].imag != 0):
        for j in range(N):
            Inverse[N-1,j] = Inverse[N-1,j] / tmp[N-1,N-1]

        tmp[N-1,N-1] = complex(1,0)

    # # ELIMINATE UPPER TRIANGLE
    for k in range(1, N):  
        for i in range(k):
            ratio = tmp[i,k] 

            for j in range(N):
                tmp[i,j] = tmp[i,j] - ratio * tmp[k,j]
                Inverse[i,j] = Inverse[i,j] - ratio * Inverse[k,j]

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
        #tr[i] = trace(array, N, tr)
        trace(array, N, tr)


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

@numba.cuda.jit()
def sqinvtz(A, B, N):
    tid = cuda.threadIdx.x
    blkid = cuda.blockIdx.x
    blkdim = cuda.blockDim.x

    i = tid + blkid * blkdim
    if(i < 1):
        inv = myInvSZ(A, B, N)