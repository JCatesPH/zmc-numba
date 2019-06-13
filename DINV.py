import math
import time

import numba
import numpy as np
import scipy
import scipy.special
from numba import cuda

import cupy as cp

start = time.time()
mu = 0.1  # Fermi-level
hOmg = 0.3  # Photon energy eV
a = 4  # AA
A = 4  # hbar^2/(2m)=4 evAA^2 (for free electron mass)
rati = 0.3  # ratio
eE0 = rati * ((hOmg) ** 2) / (2 * np.sqrt(A * mu))
# print(eE0)
Gamm = 0.003  # Gamma in eV.
KT = 1 * 10 ** (-6)
shift = A * (eE0 / hOmg) ** 2
V0 = eE0 * A / hOmg
V2 = A * (eE0 / hOmg) ** 2

d = np.zeros((3,3))
# Example: Creating a tri-diagonal matrix with no for loops
# N = 3
# xx = np.arange(-(N - 1) / 2, (N - 1) / 2 + 1, 1)
# dd = np.diag(xx)
# x = np.eye(3, 3, k=-1) + np.eye(3, 3) * 2 + np.eye(3, 3, k=1) * 3

# tstexp = np.exp(x)
# inx = np.linalg.inv(x)

# res = np.matmul(inx, x)

@numba.cuda.jit()
def ds(kx, ky, qx, qy, om, d):
    N = 3

    topkq = -complex(0, 1) * V0 * ((kx + qx) - complex(0, 1) * (ky + qy))
    botkq = complex(0, 1) * V0 * ((kx + qx) + complex(0, 1) * (ky + qy))
    innkq = om + complex(0, 1) * Gamm - A * ((kx + qx) ** 2 + (ky + qy) ** 2) - V2

    topk = -complex(0, 1) * V0 * (kx - complex(0, 1) * ky)
    botk = complex(0, 1) * V0 * (kx + complex(0, 1) * ky)
    innk = om + complex(0, 1) * Gamm - A * (kx ** 2 + ky ** 2) - V2


#    cent = cp.arange(-(N - 1) / 2, (N - 1) / 2 + 1, 1)
    cent = cp.arange(3)

    d = hOmg * cp.diag(cent, dtype=cp.float32)
    
    return d

'''
    Ginkq = np.eye(N, N, k=1) * topkq + np.eye(N, N, k=-1) * botkq + innkq * np.eye(N, N) - d

    Gink = np.eye(N, N, k=1) * topk + np.eye(N, N, k=-1) * botk + innk * np.eye(N, N) - d

    Grkq = np.linalg.inv(Ginkq)
    Gakq = np.transpose(np.conj(Grkq))

    Grk = np.linalg.inv(Gink)
    Gak = np.transpose(np.conj(Grk))

    fer = np.heaviside(-(d + np.eye(N, N) * (om - mu)), 0)

    in1 = np.matmul(Grkq, np.matmul(Grk, np.matmul(fer, Gak)))
    in2 = np.matmul(Grkq, np.matmul(fer, np.matmul(Gakq, Gak)))
    tr = np.trace(in1 + in2)
    # HERE i will divide by DOS, multiply by 2 for spin, and divide by (2pi)^3

    dchi = -(4) * Gamm * tr / math.pi ** 2

    return dchi
'''
    
test = ds[1,32](0.1, 0.1, 0.01, 0, 0.09,d)
result = cp.asnumpy(test)
end = time.time()
print('Time:',end - start)
print(result)
