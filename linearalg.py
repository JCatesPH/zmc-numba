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
N = 3

A = np.array([[2,2,2],[3,3,3],[5,5,5]])
B = np.array([[1,0,0],[0,1,0],[0,0,1]])

C = np.zeros((3,3))

print('C before:\n', C)

print('\nA:\n', A)
print('B:\n', B)

C = squareMatMul(A, B, C, N)

print('\nC=\n', C)

#%%
N = 11
trials = 25

timarr = np.zeros(trials)
stdarr = np.zeros(trials)
comparr = np.zeros(trials)

for x in range(trials):
    A = np.random.randint(-50, 50, (N, N))
    B = np.random.randint(-50, 50, (N, N))

    C = np.zeros((N, N))

    tic = time.time()
    C = squareMatMul(A, B, C, N)
    toc = time.time()

    timarr[x] = toc - tic

    tic = time.time()
    test = np.matmul(A,B)
    toc = time.time()

    stdarr[x] = toc - tic

    if (test==C).all():
        comparr[x] = True

    else:
        comparr[x] = False


print('\nN =', N)
print('============')
print(' Time diff ')
print('============')
for k in range(0,trials):
    print('%8.5f ' % (timarr[k]))
    print(comparr[k])
print('============')
print('Average: %5.3E'% (timarr.sum()/trials))

#%%
