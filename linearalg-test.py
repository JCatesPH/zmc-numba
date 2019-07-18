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
print(iden)


np.set_printoptions(precision=4, suppress=True)

#%%


Gink = np.eye(N, N, k=1) * topk + np.eye(N, N, k=-1) * botk + innk * np.eye(N, N) - d
Gk = la.invZTmat(N, botd, innd, topd, iden)

print('Gink = \n', Gink)
print('Gk = \n', Gk)

b = np.matmul(Gk, Gink)
print('\nb = \n', b)

print('or is it...')

b = np.matmul(Gink, Gk)
print('b = \n', b)

print('\nnumpy result: \n', np.linalg.inv(Gink))

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

#%%
N = 3

A = np.array([[2,2,2],[3,3,3],[5,5,5]])
B = np.array([[1,0,0],[0,1,0],[0,0,1]])

C = np.zeros((3,3))

print('C before:\n', C)

print('\nA:\n', A)
print('B:\n', B)

C = squareMatMul(A, B, C, N)

print('\nC=\n', C)

#%%
#####################################################################
# # MATRIX MULT TESTING
#####################################################################
N = 11
trials = 25

timarr = np.zeros(trials)
stdarr = np.zeros(trials)
comparr = np.zeros(trials)

for x in range(trials):
    A = np.random.randint(-50, 50, (N, N))
    B = np.random.randint(-50, 50, (N, N))

    C = np.zeros((N, N))

    tic = time.time()
    C = squareMatMul(A, B, C, N)
    toc = time.time()

    timarr[x] = toc - tic

    tic = time.time()
    test = np.matmul(A,B)
    toc = time.time()

    stdarr[x] = toc - tic

    if (test==C).all():
        comparr[x] = True

    else:
        comparr[x] = False


print('\nN =', N)
print('============')
print(' Time diff ')
print('============')
for k in range(0,trials):
    print('%8.5f ' % (timarr[k]))
    print(comparr[k])
print('============')
print('Average: %5.3E'% (timarr.sum()/trials))