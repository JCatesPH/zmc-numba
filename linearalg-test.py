#%%
import numpy as np
import linearalg as la

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

kx = 0.1
ky = 0.1
qx = 0.1
qy = 0 
om = 0.09

N = 3

topk = -complex(0, 1) * V0 * (kx - complex(0, 1) * ky)
botk = complex(0, 1) * V0 * (kx + complex(0, 1) * ky)
innk = om + complex(0, 1) * Gamm - A * (kx ** 2 + ky ** 2) - V2

topd = topk * np.ones(N-1, dtype=np.complex)
botd = topk * np.ones(N-1, dtype=np.complex)
innd = topk * np.ones(N, dtype=np.complex)

cent = np.arange(-(N - 1) / 2, (N - 1) / 2 + 1, 1, dtype=np.complex)

d = hOmg * np.diag(cent)

iden = np.identity(N, dtype=np.complex)

Gink = np.eye(N, N, k=1) * topk + np.eye(N, N, k=-1) * botk + innk * np.eye(N, N) - d

np.set_printoptions(precision=4, suppress=True)

#%%
Gk = la.invZTmat(N, botd, innd, topd, iden)

print('Gink = \n', Gink)
print('Gk = \n', Gk)

b = np.matmul(Gk, Gink)
print('b = \n', b)

print('or is it...')

b = np.matmul(Gink, Gk)
print('b = \n', b)

#%%

test = np.arange(0,3)

print(test)
print(test[1])

#%%
N = 3
for m in range(0,N-1):
    print(m)
for k in range(N-3, -1, -1):
    print(k)
