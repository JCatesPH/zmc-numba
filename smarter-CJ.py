import numpy as np
from numba import cuda


@cuda.jit()
def smartH(A, A_dag, N):
    row, col = cuda.grid(2)

    if row < N and col < N:
        A_dag[row, col] = complex(A[col,row].real, -A[col,row].imag)
        #A_dag[row, col] = A[row, col]


N = 5

top = np.array([2+2j, -4-4j, 8+4j, 4-2j])
bot = np.array([2+2j,  2-2j, 2+1j, 4-2j])
inn = np.array([2+2j,  2-2j, 2-2j, 8+2j, 4+1j])

A = np.diag(top, k=1) + np.diag(bot, k=-1) + np.diag(inn)
print('A = \n', A)

contran = np.ones_like(A)

threadsperblock = 32
blockspergrid = (A.size + (threadsperblock - 1)) // threadsperblock

smartH[blockspergrid, (threadsperblock, threadsperblock)](A, contran, N)


print('A = \n', A)
print('A* = \n', contran)
