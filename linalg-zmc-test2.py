import math
import time
import numba
import ZMCIntegral
import linearalgcuda as la
import numpy as np

N = 2
#I = np.eye(N, dtype=np.complex64)
#B = numba.cuda.to_device(I)

# user defined function
@numba.cuda.jit(device=True)
def my_func(x):
    # Declare empty arrays to be filled with the values passed into the function 
    # and the inverse of A into B
    A = numba.cuda.shared.array((N,N), dtype=numba.types.complex64)
    B = numba.cuda.shared.array((N,N), dtype=numba.types.complex64)

    for i in range(N):
        for j in range(N):
            if i == j:
                B[i,j] = complex(1,0)
            else:
                B[i,j] = complex(0,0)

    # Assign the values in the array
    A[0, 0] = math.cos(x[0])
    A[0, 1] = complex(math.cos(x[1]), math.sin(x[2]))
    A[1, 0] = complex(math.cos(x[1]), -math.sin(x[2]))
    A[1, 1] = math.cos(x[3])

    #B[0, 0] = complex(1, 0)
    #B[0, 1] = complex(0, 0)
    #B[1, 0] = complex(0, 0)
    #B[1, 1] = complex(1, 0)

    #la.myInvSZ(A, B, N)

    # # ELIMINATE LOWER TRIANGLE
    for k in range(N-1):
        #diag = A[k,k]
        
        for i in range(k+1, N):
            #ratio =  A[i,k] / A[k,k]

            for j in range(N):
                B[i,j] = B[i,j] - A[i,k] / A[k,k] * B[k,j]
                A[i,j] = A[i,j] - A[i,k] / A[k,k] * A[k,j]

    # # ELIMINATE UPPER TRIANGLE
    for k in range(N-1, 0, -1):
        #diag = A[k,k]
        
        for i in range(k-1, -1, -1):
            #ratio = A[i,k] / A[k,k]

            for j in range(N):
                B[i,j] = B[i,j] - A[i,k] / A[k,k] * B[k,j]
                A[i,j] = A[i,j] - A[i,k] / A[k,k] * A[k,j]

    # # REDUCE ROWS
    for i in range(N):
        #diag = A[i,i]

        for j in range(N):
            B[i,j] = B[i,j] / A[i,i]

    tr = A[0,0] + A[1,1]

    return tr.real

MC = ZMCIntegral.MCintegral(my_func,[
    [0,1],
    [2,3],
    [4,5],
    [6,7]
    ])

MC.depth = 2
MC.sigma_multiplication = 10
MC.num_trials = 5


start = time.time()
# obtaining the result
result = MC.evaluate()

end = time.time()

# print the formatted result
print('result = %s    std = %s' % (result[0], result[1]))


print('Time to calculate: %5.4f s' % (end-start))
