'''
linearalg.py declares the following linear algebra functions to be used in cuda decorated functions. 

They must all work as device functions (without CUDA kernels)
'''
#%%
import math
import time

import numba
import numpy as np



#%%
 
def myInvTZ(N, lower, diag, upper, inv):
    '''
    A CUDA device function that computes the inverse matrix for a 
        complex-valued tridiagonal matrix. From my solution on paper 
        using naive Gauss-Jordan Elimination. It now does elimination two 
        rows at a time, but the matrix must now be ODD RANK N.

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
    for i in range(1, N, 2):
        diag[i] = diag[i] - lower[i-1] / diag[i-1] * upper[i-1]
        for j in range(0, i+1):   
            inv[i,j] = inv[i,j] - lower[i-1] / diag[i-1] * inv[i-1, j]
            inv[i+1,j] = inv[i+1,j] - lower[i] / diag[i] * inv[i, j]

        diag[i+1] = diag[i+1] - lower[i] / diag[i] * upper[i]
        lower[i-1] = 0
        lower[i] = 0
    ###########################################################################
    # # Gaussian elimination of upper triangle
    ###########################################################################
    # R(i) = R(i) - a(i+1,i+2) / a(i+1,i+2) * R(i+1)
    for i in range(N-2, -1, -2):
        for j in range(0, N):
            inv[i,j] = inv[i,j] - upper[i] / diag[i+1] * inv[i+1, j]
            inv[i-1,j] = inv[i-1,j] - upper[i-1] / diag[i] * inv[i, j]

    ###########################################################################
    # # Row reduction
    ###########################################################################
    # R(i) = R(i) / a(i,i)
    for j in range(0,N):
            inv[0,j] = inv[0,j] / diag[0]
            
    for i in range(1, N, 2):
        for j in range(0,N):
            inv[i,j] = inv[i,j] / diag[i]
            inv[i+1,j] = inv[i+1,j] / diag[i+1]
    ###########################################################################

    return inv

 
def trace(arr, N):
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
    tr = 0+0j
    for i in range(0, N):
        tr = tr + arr[i,i]
    
    return tr

 
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
Lin
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

#%%
N = 3
timarr = np.zeros(20)

for k in range(0,20):
    top = np.random.rand(N-1)
    bot = np.random.rand(N-1)
    inn = np.random.rand(N)

    iden = np.identity(N, dtype=np.complex)

    A = np.diag(top, k=1) + np.diag(bot, k=-1) + np.diag(inn)

    tic1 = time.time()
    inv = np.linalg.inv(A)
    toc1 = time.time()

    tic2 = time.time()
    myInvTZ(N, bot, inn, top, iden)
    toc2 = time.time()

    print('\n', np.allclose(inv,iden))

    print('numpy time = ', toc1-tic1)
    print('my time = ', toc2-tic2)
    timarr[k] = abs((toc2-tic2)-(toc1-tic1))


print('\nN =', N)
print('============')
print(' Time diff ')
print('============')
for k in range(0,20):
    print('%8.5f ' % (timarr[k]))
print('============')
print('Average: ', timarr.sum()/20)


#%%
###########################################################################
# # Matrix Multiplication
###########################################################################
@numba.jit()
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
            for l in range(N):
                C[i,j] = A[i,l] * B[l,j] + C[i,j]
    
    return C

#%%
############S##############################################################
# # General Inverse
###########################################################################
@numba.jit()
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

        if (A[k,k].real != 1) or (A[k,k].imag != 0):
            scale = A[k,k]
            for j in range(N):
                Inverse[k,j] = Inverse[k,j] / scale
                A[k,j] = A[k,j] / scale
                
        for i in range(k+1, N):
            if (A[i,k].real != 0) or (A[i,k].imag != 0):
                ratio =  A[i,k]

                for j in range(N):
                    A[i,j] = A[i,j] - ratio * A[k,j]
                    Inverse[i,j] = Inverse[i,j] - ratio * Inverse[k,j]

    if (A[N-1,N-1].real != 1) or (A[N-1,N-1].imag != 0):
        for j in range(N):
            Inverse[N-1,j] = Inverse[N-1,j] / A[N-1,N-1]

        A[N-1,N-1] = complex(1,0)

    # # ELIMINATE UPPER TRIANGLE
    for k in range(1, N):  
        for i in range(k):
            ratio = A[i,k] 

            for j in range(N):
                A[i,j] = A[i,j] - ratio * A[k,j]
                Inverse[i,j] = Inverse[i,j] - ratio * Inverse[k,j]

    return Inverse


#%%
np.set_printoptions(precision=4, suppress=True)


N = 5

A = np.array([[5.,1,1,1,1],
              [2,6,2,2,2],
              [3,3,7,3,3],
              [4,4,4,8,4],
              [0,0,0,0,9]])  #, dtype=np.complex)
I = np.eye(N)

test = np.linalg.inv(A)

print('A:\n', A)

inv = myInvSZ(A, I, N)

print('A:\n', A)

print('\nAinv:\n', I)

print('\nTest:\n', test)
#%%


#%%
