
# coding: utf-8

# # Test to ensure modifications to function do not change evaluated value
# 
# Beginning modification of the function to handle case where N is not equal to 1.


import math
# from numba import cuda
import ZMCIntegral
import time
import numpy as np
import cudahelpers
import numba
# # Define constants in function

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

Gammsq = Gamm ** 2

@numba.cuda.jit(device=True)        
def modDsN2(x):
    N = 3
    dds = 0
    # ds = 0 # UNUSED
    qx = cudahelpers.getqx()
    ek = A * (math.sqrt((x[0]) ** 2 + (x[1]) ** 2)) ** 2 + A * (eE0 / hOmg) ** 2
    ekq = A * (math.sqrt((x[0] + qx) ** 2 + (x[1] + 0) ** 2)) ** 2 + A * (eE0 / hOmg) ** 2
    xk = 2 * A * eE0 * math.sqrt((x[0]) ** 2 + (x[1]) ** 2) / hOmg ** 2
    xkq = 2 * A * eE0 * math.sqrt((x[0] + qx) ** 2 + (x[1] + 0) ** 2) / hOmg ** 2

    singrmatrix = numba.cuda.shared.array((6,N),dtype=numba.types.f8)
    singzmatrix = numba.cuda.shared.array((4,N),dtype=numba.types.complex128)  

    n = 0
    i = -(N - 1) / 2
    for n in range(0, N):
        nu = hOmg * i
        chi = hOmg / 2
        omicron = ek - chi + nu
        phi = ekq - chi + nu
        iota = ek + chi + nu
        kappa = ekq + chi + nu

        omisq = omicron ** 2
        phisq = phi ** 2
        iotasq = iota ** 2
        kappasq = kappa ** 2

        singrmatrix[0,n] = 2 * math.atan2(Gamm, omicron)
        singrmatrix[1,n] = 2 * math.atan2(Gamm, phi)
        singrmatrix[2,n] = 2 * math.atan2(Gamm, iota)
        singrmatrix[3,n] = 2 * math.atan2(Gamm, kappa)

        singzmatrix[0,n] = complex(0, 1) * math.log(Gammsq + omisq)
        singzmatrix[1,n] = complex(0, 1) * math.log(Gammsq + phisq)
        singzmatrix[2,n] = complex(0, 1) * math.log(Gammsq + iotasq)
        singzmatrix[3,n] = complex(0, 1) * math.log(Gammsq + kappasq)

        chinu = chi - nu

        singrmatrix[4,n] = cudahelpers.my_heaviside(mu - chinu)
        singrmatrix[5,n] = cudahelpers.my_heaviside(mu + chinu)
        i = i + 1

    size_dbl = 5 # 2N-1
    dblrmatrix = numba.cuda.shared.array((5,size_dbl),dtype=numba.types.f8)
    dblzmatrix = numba.cuda.shared.array((4,size_dbl),dtype=numba.types.complex128)

    i = -(N-1)
    for n in range(0, size_dbl):
        xi = hOmg * i
        zeta = ek - mu + xi
        eta = ekq - mu + xi

        zetasq = zeta ** 2
        etasq = eta ** 2

        dblrmatrix[0,n] = 2 * math.atan2(Gamm, zeta)
        dblrmatrix[1,n] = 2 * math.atan2(Gamm, eta)

        logged1 = math.log(Gammsq + zetasq)
        logged2 = math.log(Gammsq + etasq)

        dblzmatrix[0,n] = complex(0, logged1)
        dblzmatrix[1,n] = complex(0, logged2)

        dblrmatrix[2,n] = cudahelpers.besselj(i, xk)
        dblrmatrix[3,n] = cudahelpers.besselj(i, xkq)

        fac1i = ek - ekq + xi
        fac2i = complex(fac1i, 2 * Gamm)
        dblrmatrix[4,n] = fac1i
        dblzmatrix[2,n] = fac2i
        dblzmatrix[3,n] = fac2i - ek + ekq
        i = i + 1
    
    # This is implementing what Mahmoud and I discussed to construct these outside.
    sdr00 = numba.cuda.shared.array((N,N), dtype=numba.types.f8)
    sdr20 = numba.cuda.shared.array((N,N), dtype=numba.types.f8)
    sdr11 = numba.cuda.shared.array((N,N), dtype=numba.types.f8)
    sdr31 = numba.cuda.shared.array((N,N), dtype=numba.types.f8)
    
    sdz00 = numba.cuda.shared.array((N,N), dtype=numba.types.c16)
    sdz20 = numba.cuda.shared.array((N,N), dtype=numba.types.c16)
    sdz11 = numba.cuda.shared.array((N,N), dtype=numba.types.c16)
    sdz31 = numba.cuda.shared.array((N,N), dtype=numba.types.c16)

    dd11 = numba.cuda.shared.array((N,N), dtype=numba.types.c16)
    dd00 = numba.cuda.shared.array((N,N), dtype=numba.types.c16)


    # Now to fill the arrays, we use a dummy variable that will be referenced below by the actual indices.
    for i in range(0,N):  # i will be indexed by s below
        for j in range(0,N): # j will be indexed by alpha, beta, or gamma in the nested loops
            ij = i + j
            sdr00[i,j] = singrmatrix[0,j] - dblrmatrix[0,ij]
            sdr20[i,j] = singrmatrix[2,j] - dblrmatrix[0,ij]
            sdr11[i,j] = -singrmatrix[1,j] + dblrmatrix[1,ij]
            sdr31[i,j] = -singrmatrix[3,j] + dblrmatrix[1,ij]
            
            sdz00[i,j] = singzmatrix[0,j] + dblzmatrix[0,ij]
            sdz20[i,j] = singzmatrix[2,j] + dblzmatrix[0,ij]
            sdz11[i,j] = singzmatrix[1,j] + dblzmatrix[1,ij]
            sdz31[i,j] = singzmatrix[3,j] + dblzmatrix[1,ij]

            dd11[i,j] = dblrmatrix[1,ij] - dblzmatrix[1,ij]
            dd00[i,j] = dblrmatrix[0,ij] + dblzmatrix[0,ij]



    for n in range(0, N):
        nmod = n + N - 1

        for alpha in range(0, N):

            for beta in range(0, N):
                abdiff = alpha - beta + N - 1

                for gamma in range(0, N):
                    bgdiff = beta - gamma + N - 1
                    agdiff = alpha - gamma + N - 1

                    d1 = -2 * complex(0, 1) * dblrmatrix[4,bgdiff] * dblzmatrix[2,agdiff] * dblzmatrix[3,abdiff]
                    d2 = -2 * complex(0, 1) * dblrmatrix[4,abdiff] * dblzmatrix[2,agdiff] * dblzmatrix[3,bgdiff]

                    for s in range(0, N):
                        smod = s + N - 1
                        
                        p1p = dblrmatrix[4,bgdiff] * (sdr00[s,alpha] - sdz00[s,alpha])
                        p2p = dblzmatrix[2,agdiff] * (singrmatrix[0,beta] - dd00[s,beta] + singzmatrix[0,beta])
                        p3p = dblzmatrix[3,abdiff] * (sdr11[s,gamma] - sdz11[s,gamma])

                        p1m = dblrmatrix[4,bgdiff] * (sdr20[s,alpha] - sdz20[s,alpha])
                        p2m = dblzmatrix[2,agdiff] * (singrmatrix[2,beta] - dd00[s,beta] + singzmatrix[2,beta])
                        p3m = dblzmatrix[3,abdiff] * (sdr31[s,gamma] - sdz31[s,gamma])

                        omint1p = singrmatrix[4,s] * ((p1p + p2p + p3p) / d1)
                        omint1m = singrmatrix[5,s] * ((p1m + p2m + p3m) / d1)

                        pp1p = dblrmatrix[4,abdiff] * (sdr11[s,gamma] - sdz11[s,gamma])
                        pp2p = dblzmatrix[2,agdiff] * (-singrmatrix[1,beta] + dd11[s,beta] + singzmatrix[1,beta])
                        pp3p = dblzmatrix[3,bgdiff] * (sdr00[s,alpha] - sdz00[s,alpha])

                        pp1m = dblrmatrix[4,abdiff] * (sdr31[s,gamma] - sdz31[s,gamma])
                        pp2m = dblzmatrix[2,agdiff] * (-singrmatrix[3,beta] + dd11[s,beta] + singzmatrix[3,beta])
                        pp3m = dblzmatrix[3,bgdiff] * (sdr20[s,alpha] - sdz20[s,alpha])

                        omint2p = singrmatrix[4,s] * ((pp1p + pp2p + pp3p) / d2)
                        omint2m = singrmatrix[5,s] * ((pp1m + pp2m + pp3m) / d2)

                        for l in range(0, N):
                            lmod = l + N - 1

                            bess1 = dblrmatrix[3,gamma - nmod] * dblrmatrix[3,gamma - lmod] * dblrmatrix[2,beta - lmod] * dblrmatrix[2,beta - smod] * dblrmatrix[2,alpha - smod] * dblrmatrix[2,alpha - nmod]

                            grgl = bess1 * (omint1p - omint1m)

                            bess2 = dblrmatrix[3,gamma - nmod] * dblrmatrix[3,gamma - smod] * dblrmatrix[3,beta - smod] * dblrmatrix[3,beta - lmod] * dblrmatrix[2,alpha - lmod] * dblrmatrix[2,alpha - nmod]

                            glga = bess2 * (omint2p - omint2m)

                            dds = dds + Gamm * (grgl + glga)
    return -4 * dds.real / math.pi ** 2



tic = time.time()

depths = 2
sigmults = 1E4
trials = 5

print('Following values are constant for all integrations.')
print('\n========================================================')
print('\ndepth = ', depths)
print('sigma_multiplication = ', sigmults)
print('num_trials = ', trials)
print('available_GPU = [0]')
print('kxi = - math.pi / a')
print('kxf = math.pi / a')
print('kyi = - math.pi / a')
print('kyf = math.pi / a')
print('\n========================================================')

# qx = getqx()

# SET PARAMETERS AND LIMITS OF INTEGRATION
kxi = - math.pi / a
kxf = math.pi / a

kyi = - math.pi / a
kyf = math.pi / a

# qx = [0.001,6pi/a]

resultArr = np.zeros(100)
errorArr = np.zeros(100)
timeArr = np.zeros(100)
j = 0
for i in np.linspace(.01, .785, 100):

    cudahelpers.setqx(i)
    MC = ZMCIntegral.MCintegral(modDsN2,[[kxi,kxf],[kyi,kyf]])
    # Setting the zmcintegral parameters
    MC.depth = depths
    MC.sigma_multiplication = sigmults
    MC.num_trials = trials
    start = time.time()
    result = MC.evaluate()
    print('Result for qx = ',i, ': ', result[0], ' with error: ', result[1])
    print('================================================================')
    end = time.time()
    print('Computed in ', end-start, ' seconds.')
    print('================================================================')
    resultArr[j] = result[0]
    errorArr[j] = result[1]
    timeArr[j] = end - start
    j = j + 1




print('================================================================')


j = 0
print('All values in csv format:')
for i in np.linspace(.01, .785, 100):
    print('%5.3f, %11.8E, %5.3E, %5.3E' % (i, resultArr[j], errorArr[j], timeArr[j]))
    j = j + 1

toc = time.time()
print('================================================================\n')
print('Process completed successfully!')
print('Total time is ', toc-tic, 'seconds.')
