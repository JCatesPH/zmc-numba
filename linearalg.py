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
