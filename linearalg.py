'''
linearalg.py declares the following linear algebra functions to be used in cuda decorated functions. 

They must all work as device functions (without CUDA kernels)
'''
#%%
def invZTmat(N, lower, diag, upper, inv):
    '''
    Computes the inverse matrix for a complex-valued tridiagonal matrix.
    This is:  A * A**-1 = I
        A : Tridiagonal, N x N matrix
        A**-1 : Inverse of A
        I : N x N Identity matrix

    Translated and adapted from LAPACK file cgtsv.f, which is found at http://www.netlib.org/lapack/explore-html/d3/dc4/cgtsv_8f_source.html

    INPUT:
        N : (int) Size of square matrix
        lower : (complex array) Vector of size (N-1) that is the lower diagonal entries of A
        diag : (complex array) Vector of size N that is the diagonal entries of A
        upper : (complex array) Vector of size (N-1) that is the upper diagonal entries of A
        inv : N x N Identity matrix

    OUTPUT:
        inv : (complex matrix) Matrix of size N x N that is the inverse of A
    '''

    mult = complex(1,1)
    temp = complex(1,1)

    for k in range(0, N-1):

        # Checks if no row interchange is required
        if(abs(diag[k].real) >= abs(lower[k].real) and abs(diag[k].imag) >= abs(lower[k].imag)):
                
            mult = lower[k] / diag[k]
            diag[k+1] = diag[k+1] - mult * upper[k]

            for j in range(0, N):
                inv[k+1, j] = inv[k+1, j] - mult * inv[k, j]
            

            if(k < (N-2)):
                lower[k] = complex(0,0)


        else:

            mult = diag[k] / lower[k]
            diag[k] = lower[k]
            temp = diag[k+1]
            diag[k+1] = upper[k] - mult * temp

            if(k < (N-2)):
                lower[k] = upper[k+1]
                upper[k+1] = -mult * lower[k]
            
            upper[k] = temp
            
            for j in range(0, N):
                temp = inv[k, j]
                inv[k, j] = inv[k+1, j]
                inv[k+1, j] = temp - mult * inv[k+1, j]
        

    if(diag[N-1] == 0.+0j):
        return inv

    # Back solve with matrix U from the factorization
    for j in range(0, N):
        inv[N-1, j] = inv[N-1, j] / diag[N-1]

        if(N > 1):
            inv[N-2, j] = (inv[N-2, j] - upper[N-2] * inv[N-1, j]) / diag[N-2]

        for k in range(N-3, -1, -1):
            inv[k, j] = (inv[k, j] - upper[k] * inv[k+1, j] - lower[k] * inv[k+2, j]) / diag[k]
    
    return inv




#%%
def myInvTZN3(N, lower, diag, upper, inv):
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
    # R2 = R2 - a21 / a11 * R1   for I
    for i in range(0, N):
        inv[1,i] = inv[1,i] - lower[0] / diag[0] * inv[0, i]
    
    # R2 = R2 - a21 / a11 * R1   for A
    diag[1] = diag[1] - lower[0] / diag[0] * upper[1]
    lower[0] = 0

    # R3 = R3 - a32 / a22 * R2   for I
    for i in range(0,N):
        inv[2,i] = inv[2,i] - lower[1] / diag[1] * inv[1, i]

    # R3 = R3 - a32 / a22 * R2   for A
    diag[2] = diag[2] - lower[1] / diag[1] * upper[2]
    lower[1] = 0
    
    ###########################################################################
    # # Gaussian elimination of upper triangle
    ###########################################################################
    # R2 = R2 - a23 / a33 * R3   for I
    for i in range(0, N):
        inv[1,i] = inv[1,i] - upper[2] / diag[2] * inv[2, i]

    # R2 = R2 - a23 / a33 * R3   for A
    upper[2] = 0

    # R1 = R1 - a12 / a22 * R2   for I
    for i in range(0, N):
        inv[0,i] = inv[0,i] - upper[1] / diag[1] * inv[1, i]
    
    # R1 = R1 - a12 / a22 * R2   for A
    upper[1] = 0

    ###########################################################################
    # # Row reduction
    ###########################################################################
    # R1 / a11, R2 / a22, R3 / a33
    for j in range(0,N):
        inv[0,j] = inv[0,j] / diag[0]
        inv[1,j] = inv[1,j] / diag[1]
        inv[2,j] = inv[2,j] / diag[2]

    return inv

#%%
import numpy as np

N = 3

top = np.array([0, 2+2j, -4-4j])
bot = np.array([2+2j, 2-2j, 0])
inn = np.array([2+2j, 2-2j, 2-2j])

iden = np.identity(3, dtype=np.complex)

A = np.eye(N, N, k=1) * top + np.eye(N, N, k=-1) * bot + inn * np.eye(N, N)

print('A = \n', A)

np.set_printoptions(precision=4, suppress=True)

Ainv = myInvTZN3(N, bot, inn, top, iden)

print('Ainv = \n', Ainv)

#b = np.matmul(A, Ainv)
#print('\nb = \n', b)

#print('or is it...')

b = np.matmul(A, Ainv)
print('b = \n', b)

print('\nnumpy result: \n', np.linalg.inv(Ainv))

#%%
