'''
linearalg.py declares the following linear algebra functions to be used in cuda decorated functions. 

They must all work as device functions (without CUDA kernels)
'''
#%%
def invZTmat(N, DL, D, DU, X):
    '''
    Computes the inverse matrix for a complex-valued tridiagonal matrix.
    This is:  A * A**-1 = I
        A : Tridiagonal, N x N matrix
        A**-1 : Inverse of A
        I : N x N Identity matrix

    Translated and adapted from LAPACK file cgtsv.f, which is found at http://www.netlib.org/lapack/explore-html/d3/dc4/cgtsv_8f_source.html

    INPUT:
        N : (int) Size of square matrix
        DL : (complex array) Vector of size (N-1) that is the lower diagonal entries of A
        D : (complex array) Vector of size N that is the diagonal entries of A
        DU : (complex array) Vector of size (N-1) that is the upper diagonal entries of A
        X : N x N Identity matrix

    OUTPUT:
        X : (complex matrix) Matrix of size N x N that is the inverse of A
    '''

    mult = complex(1,1)
    temp = complex(1,1)

    for k in range(0, N-1):
        # Checks if no row interchange is required
        if(abs(D[k].real) >= abs(DL[k].real) and abs(D[k].imag) >= abs(DL[k].imag)):
            mult = DL[k] / D[k]
            D[k+1] = D[k+1] - mult * DU[k]

            for j in range(0, N):
                X[j, k+1] = X[j, k+1] - mult * X[j, k]
            
            if(k < (N-2)):
                DL[k] = complex(0,0)

        # Interchange rows k and k+1
        else:
            mult = D[k] / DL[k]
            D[k] = DL[k]
            temp = D[k+1]
            D[k+1] = DU[k] - mult * temp

            if(k < (N-2)):
                DL[k] = DU[k+1]
                DU[k+1] = -mult * DL[k]
            
            DU[k] = temp

            for j in range(0, N):
                temp = X[j, k]
                X[j, k] = X[j, k+1]
                X[j, k+1] = temp - mult * X[j, k+1]
    
    if(D[N-1] == 0.+0j):
        return X

    # Back solve with matrix U from the factorization
    for j in range(0, N):
        X[j, N-1] = X[j, N-1] / D[N-1]

        if(N > 1):
            X[j, N-2] = (X[j, N-2] - DU[N-2] * X[j, N-1]) / D[N-2]
        
        for k in range(N-3, -1, -1):
            X[j, k] = (X[j, k] - DU[k] * X[j, k+1] - DL[k] * X[j, k+2]) / D[k]
    
    return X



#%%
N = 3
for j in range(0,N-1):
    print(j)


#%%
