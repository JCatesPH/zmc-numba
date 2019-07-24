#%%
import math
import time
import numba
from numba import cuda
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
    A[0, 0] = 1 / complex(x[0], x[1])
    A[0, 1] = complex(0, 0)
    A[1, 0] = complex(0, 0)
    A[1, 1] = 1 / complex(x[2], x[3])

    for i in range(N):
        for j in range(N):
            if i == j:
                B[i,j] = complex(1,0)
        
            else:
                B[i,j] = complex(0,0)

    # B = la.myInvSZ(A, B, N)

    for k in range(N-1):

        if (A[k,k].real != 1) or (A[k,k].imag != 0):
            scale = A[k,k]
            for j in range(N):
                B[k,j] = B[k,j] / scale
                A[k,j] = A[k,j] / scale
                
        for i in range(k+1, N):
            if (A[i,k].real != 0) or (A[i,k].imag != 0):
                ratio =  A[i,k]

                for j in range(N):
                    A[i,j] = A[i,j] - ratio * A[k,j]
                    B[i,j] = B[i,j] - ratio * B[k,j]

    if (A[N-1,N-1].real != 1) or (A[N-1,N-1].imag != 0):
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

    return (tr * tr.conjugate()).real

#%%
@numba.cuda.jit()
def kernelfunc(points, results, samples):
    tid = cuda.threadIdx.x
    blkid = cuda.blockIdx.x
    blkdim = cuda.blockDim.x

    i = tid + blkid * blkdim
    if(i < samples):
        results[i] = my_func(points[i])


#%%
samples = 100000

a1 = 1; b1 = 2
x = (b1 - a1) * np.random.random_sample((samples,1)) + a1
a2 = 2; b2 = 3
y = (b2 - a2) * np.random.random_sample((samples,1)) + a2
a3 = 3; b3 = 4
z = (b3 - a3) * np.random.random_sample((samples,1)) + a3
a4 = 4; b4 = 5
q = (b4 - a4) * np.random.random_sample((samples,1)) + a4

matrix = np.concatenate([x, y, z, q], axis=1)
results = np.zeros(samples)

threadsperblock = 32
blockspergrid = (samples + (threadsperblock - 1)) // threadsperblock

kernelfunc[blockspergrid, threadsperblock](matrix, results, samples)

norm = (b1-a1) * (b2-a2) * (b3-a3) * (b4-a4)
Q_N = norm / samples * np.sum(results)
Q_err = np.var(results) / samples

#%%
# print the formatted result
print('result = {:.4E}, error?= {:.3E}'.format(Q_N, Q_err))

# print('Time to calculate: %5.4f s' % (end-start))


#%%
