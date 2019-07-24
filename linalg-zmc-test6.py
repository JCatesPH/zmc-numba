#%%
import math
import time
import numba
import ZMCIntegral
import linearalgcuda as la
import numpy as np

N = 2
#I = np.eye(N, dtype=np.complex64)
#B = numba.cuda.to_device(I)

#%%
# user defined function
@numba.cuda.jit(device=True)
def my_func(x):
    # Declare empty arrays to be filled with the values passed into the function 
    # and the inverse of A into B
    A = numba.cuda.shared.array((N,N), dtype=numba.types.complex64)
    B = numba.cuda.shared.array((N,N), dtype=numba.types.complex64)
    #tmp = numba.cuda.shared.array((N,N), dtype=numba.types.complex64)

    # Assign the values in the array
    A[0, 0] = complex(1 / x[0], 0)
    A[0, 1] = complex(x[1], x[2])
    A[1, 0] = complex(x[1], -x[2])
    A[1, 1] = complex(1 / x[3], 0)

    for i in range(N):
        for j in range(N):
            if i == j:
                B[i,j] = complex(1,0)
        
            else:
                B[i,j] = complex(0,0)

    # B = la.myInvSZ(A, B, N)

    for k in range(N-1):
        scale = A[k,k]
        for j in range(N):
            B[k,j] = B[k,j] / scale
            A[k,j] = A[k,j] / scale
                
        for i in range(k+1, N):
            ratio =  A[i,k]

            for j in range(N):
                A[i,j] = A[i,j] - ratio * A[k,j]
                B[i,j] = B[i,j] - ratio * B[k,j]


    for j in range(N):
        B[N-1,j] = B[N-1,j] / A[N-1,N-1]

    A[N-1,N-1] = complex(1,0)

    # # ELIMINATE UPPER TRIANGLE
    for k in range(1, N):  
        for i in range(k):
            ratio = A[i,k] 

            for j in range(N):
                A[i,j] = A[i,j] - ratio * A[k,j]
                B[i,j] = B[i,j] - ratio * B[k,j]

    tr = 0 + 0j
    tr = la.trace(B, N, tr)

    return tr.real # (tr * tr.conjugate()).real

#%%
MC = ZMCIntegral.MCintegral(my_func,[
    [1,2],
    [2,3],
    [3,4],
    [4,5]
    ])

MC.depth = 2
MC.sigma_multiplication = 10
MC.num_trials = 8

#%%
start = time.time()
# obtaining the result
result = MC.evaluate()

end = time.time()

#%%
# print the formatted result
print('result = %s    std = %s' % (result[0], result[1]))


print('Time to calculate: %5.4f s' % (end-start))

