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
    C = numba.cuda.shared.array((N,N), dtype=numba.types.complex64)

    # Assign the values in the array
    A[0, 0] = complex(x[0], x[1])
    A[0, 1] = complex(x[2], x[3])
    A[1, 0] = complex(-x[2], x[3])
    A[1, 1] = complex(x[0], -x[1])

    B[0, 0] = complex(x[0], -x[1])
    B[1, 0] = complex(x[2], -x[3])
    B[0, 1] = complex(-x[2], -x[3])
    B[1, 1] = complex(x[0], x[1])

    C = la.squareMatMul(A, B, C, N)

    #for i in range(N):
    #
    #    for j in range(N):
    #        tmp = 0+0j
    #
    #        for l in range(N):
    #            tmp = A[i,l] * B[l,j] + tmp
    #
    #        C[i,j] = tmp

    # tr = C[0,0] + C[1,1]

    tr = 0 + 0j

    tr = la.trace(C, N, tr)

    return tr.real

MC = ZMCIntegral.MCintegral(my_func,[
    [0,1],
    [0,1],
    [4,5],
    [6,7]
    ])

MC.depth = 2
MC.sigma_multiplication = 10
MC.num_trials = 7


start = time.time()
# obtaining the result
result = MC.evaluate()

end = time.time()

# print the formatted result
print('''
A =
[[a + i b,  c + i d],
[-c + i d,  a - i b]]
B =
[[a - i b, -c - i d],
[ c - i d,  a + i b]]

Limits:
    [0,1],
    [0,1],
    [4,5],
    [6,7]

''')

print('Testing Multiplication: ')

print('result = %s    std = %s' % (result[0], result[1]))

print('Time to calculate: %5.4f s' % (end-start))

