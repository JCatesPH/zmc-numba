#%%
import time
import numpy as np
import linearalg as la

#%%
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
Gk = la.myInvZT(N, botd, innd, topd, iden)

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

C = la.squareMatMul(A, B, C, N)

print('\nC=\n', C)

#%%
#####################################################################
# # MATRIX MULT TESTING
#####################################################################
N = 11
trials = 100

timarr = np.zeros(trials)
stdarr = np.zeros(trials)
comparr = np.zeros(trials)

for x in range(trials):
    A = np.random.randint(-50, 50, (N, N))
    B = np.random.randint(-50, 50, (N, N))

    C = np.zeros((N, N))

    tic = time.time()
    C = la.squareMatMul(A, B, C, N)
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


diffarr = timarr - stdarr

print('\nN =', N)
print('============')
print(' Time diff ')
print('============')
for k in range(0,trials):
    print('%8.5f | %r' % (timarr[k], comparr[k]))
    #print(comparr[k])
print('============')
print('Average: %5.3E'% (diffarr.sum()/trials))

#%%
#####################################################################
# # GENERAL MATRIX INVERSION
#####################################################################
N = 5
trials = 100

timarr = np.zeros(trials)
stdarr = np.zeros(trials)
comparr = np.zeros(trials)

A = np.random.rand(N,N)
I = np.eye(N)

inv = la.myInvSZ(A, I, N)


for x in range(trials):
    A = np.random.rand(N, N)

    I = np.eye(N)

    tic = time.time()
    test = np.linalg.inv(A)
    toc = time.time()

    stdarr[x] = toc - tic


    tic = time.time()
    C = la.myInvSZ(A, I, N)
    toc = time.time()

    timarr[x] = toc - tic


    if (test==C).all():
        comparr[x] = True

    else:
        comparr[x] = False

diffarr = timarr - stdarr

print('\nN =', N)
print('============')
print(' Time diff ')
print('============')
for k in range(0,trials):
    print('%8.5E | %r' % (diffarr[k], comparr[k]))
    #print(comparr[k])
print('============')
print('Average: %5.3E'% (diffarr.sum()/trials))

#%%
# # # # # # # # # # # 
# TESTING NEW INV   #
# # # # # # # # # # # 
trials = 10000
N = 5

a = -1000.
b = 1000.

testarr = np.zeros(trials)
timearr = np.zeros(trials)

for i in range(trials):
    A_re = (b - a) * np.random.random_sample((N, N)) + a
    A_im = (b - a) * np.random.random_sample((N, N)) + a

    A = np.empty((N,N), dtype=np.complex64)
    A = A_re + A_im * 1j

    I = np.eye(N, dtype=np.complex64)

    tic = time.time()
    inv_np = np.linalg.inv(A)
    toc = time.time()
    nptime = toc - tic

    tic = time.time()
    inv_test = myInvSZ(A, I, N)
    toc = time.time()
    mytime = toc - tic

    testarr[i] = np.allclose(inv_np, inv_test)
    timearr[i] = mytime - nptime

    #print('\nTime: ', mytime)
    #print('Same: ', testarr[i])


print('Average time diff= ', timearr.mean())
print('All equal: ', testarr.all())

#%%
#####################################################################
# # CUDA CALLS
#####################################################################
import linearalgcuda as la
import time
import numpy as np

#%%
# # # # # # # # # # # 
# TESTING NEW TRACE #
# # # # # # # # # # # 

A = np.array(
    [[2, 4],
     [3, 5]],
     dtype=np.complex64
)

N = 2
tr = np.array([0], dtype=np.complex64)

la.tktr[1, 32](A, N, tr)
print('tr = ', tr[0])

##########################################################################
# # # # # # # # # # # 
# TRIDIAG INVERSE   #
# # # # # # # # # # # 
N = 5


top = np.array([2+2j, -4-4j, 8+4j, 4-2j])
bot = np.array([2+2j,  2-2j, 2+1j, 4-2j])
inn = np.array([2+2j,  2-2j, 2-2j, 8+2j, 4+1j])

iden = np.identity(N, dtype=np.complex)

A = np.diag(top, k=1) + np.diag(bot, k=-1) + np.diag(inn)
print('A = \n', A)

np.set_printoptions(precision=4, suppress=True, linewidth=90)

la.tkinvtz[1, 32](N, bot, inn, top, iden)
print('Ainv = \n', iden)

##########################################################################

b = np.matmul(A, iden)
print('b = \n', b)

print('\nnumpy result: \n', np.linalg.inv(A))

##########################################################################
# # # # # # # # # # # 
# TESTING CONJ-TRAN #
# # # # # # # # # # # 
contran = np.ones((N,N), dtype=np.complex)
la.tkcj[1, 32](A, N, contran)
print('A = \n', A)
print('A* = \n', contran)

##########################################################################
# # # # # # # # # # # 
# TRIDIAG INVERSE   #
# # # # # # # # # # # 
N = 1000

for k in range(0,10):
    top = np.random.rand(N-1)
    bot = np.random.rand(N-1)
    inn = np.random.rand(N)

    iden = np.identity(N, dtype=np.complex)

    A = np.diag(top, k=1) + np.diag(bot, k=-1) + np.diag(inn)

    tic1 = time.time()
    inv = np.linalg.inv(A)
    toc1 = time.time()

    tic2 = time.time()
    la.tkinvtz[1, 32](N, bot, inn, top, iden)
    toc2 = time.time()

    print(np.isclose(inv,iden))

    print('numpy time = ', toc1-tic1)
    print('my time = ', toc2-tic2)



##########################################################################
# # # # # # # # # # # 
# MATRIX MULT TEST  #
# # # # # # # # # # # 
N = 7

mat1 = np.random.randint(-50, 50, (N,N))
mat2 = np.random.randint(-50, 50, (N,N))

res = np.zeros((N,N))

la.gsmm[1, 32](mat1, mat2, res, N)

test = np.matmul(mat1,mat2)

print(res)

print('\nSame:', (test==res).all())

#%%
##########################################################################
# # # # # # # # # # # 
# TESTING NEW INV   #
# # # # # # # # # # # 
trials = 10
N = 5

a = -1000.
b = 1000.

testarr = np.zeros(trials)
timearr = np.zeros(trials)

for i in range(trials):
    A_re = (b - a) * np.random.random_sample((N, N)) + a
    A_im = (b - a) * np.random.random_sample((N, N)) + a

    A = np.empty((N,N), dtype=np.complex128)
    A = A_re + A_im * 1j

    I = np.eye(N, dtype=np.complex128)

    tic = time.time()
    inv_np = np.linalg.inv(A)
    toc = time.time()
    nptime = toc - tic

    tic = time.time()
    la.sqinvtz(A, I, N)
    toc = time.time()
    mytime = toc - tic

    testarr[i] = np.mean(inv_np - I)
    timearr[i] = mytime - nptime

    #print('\nTime: ', mytime)
    #print('Same: ', testarr[i])


print('Average time diff= ', timearr.mean())
print('All equal: ', testarr.all())


#%%
