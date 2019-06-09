
# coding: utf-8

# # Test to ensure modifications to function do not change evaluated value
# 
# Beginning modification of the function to handle case where N is not equal to 1.

#%%
import math
from numba import cuda
import ZMCIntegral
import time
import numpy as np
import cudahelpers as helpers
import numba
import originalscript as mma
# # Define constants in function

mu = 0.1  # Fermi-level
hOmg = 0.3  # Photon energy eV
a = 4  # AA
A = 4  # hbar^2/(2m)=4 evAA^2 (for free electron mass)
rati = 0.3 # ratio
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
    qx = helpers.getqx()
    ek = A * (math.sqrt((x[0]) ** 2 + (x[1]) ** 2)) ** 2 + A * (eE0 / hOmg) ** 2
    ekq = A * (math.sqrt((x[0] + qx) ** 2 + (x[1] + 0) ** 2)) ** 2 + A * (eE0 / hOmg) ** 2
    xk = 2 * A * eE0 * math.sqrt((x[0]) ** 2 + (x[1]) ** 2) / hOmg ** 2
    xkq = 2 * A * eE0 * math.sqrt((x[0] + qx) ** 2 + (x[1] + 0) ** 2) / hOmg ** 2

    singmatrix = numba.cuda.shared.array((10,N),dtype=numba.types.complex128)  

    n = 0
    i = -(N - 1) / 2
    while(i < ((N - 1) / 2 + 1)):
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

        singmatrix[0,n] = 2 * math.atan2(Gamm, omicron)
        singmatrix[1,n] = 2 * math.atan2(Gamm, phi)
        singmatrix[2,n] = 2 * math.atan2(Gamm, iota)
        singmatrix[3,n] = 2 * math.atan2(Gamm, kappa)

        singmatrix[4,n] = complex(0, 1) * math.log(Gammsq + omisq)
        singmatrix[5,n] = complex(0, 1) * math.log(Gammsq + phisq)
        singmatrix[6,n] = complex(0, 1) * math.log(Gammsq + iotasq)
        singmatrix[7,n] = complex(0, 1) * math.log(Gammsq + kappasq)

        chinu = chi - nu

        singmatrix[8,n] = helpers.my_heaviside(mu - chinu)
        singmatrix[9,n] = helpers.my_heaviside(mu + chinu)
        i = i + 1
        n = n + 1


    size_dbl = 5
    dblmatrix = numba.cuda.shared.array((9,size_dbl),dtype=numba.types.complex128)

    n = 0
    for i in range(-(N - 1), N, 1):
        xi = hOmg * i
        zeta = ek - mu + xi
        eta = ekq - mu + xi

        zetasq = zeta ** 2
        etasq = eta ** 2

        dblmatrix[0,n] = 2 * math.atan2(Gamm, zeta)
        dblmatrix[1,n] = 2 * math.atan2(Gamm, eta)

        logged1 = math.log(Gammsq + zetasq)
        logged2 = math.log(Gammsq + etasq)

        dblmatrix[2,n] = complex(0, logged1)
        dblmatrix[3,n] = complex(0, logged2)

        dblmatrix[4,n] = helpers.besselj(i, xk)
        dblmatrix[5,n] = helpers.besselj(i, xkq)

        fac1i = ek - ekq + xi
        fac2i = complex(fac1i, 2 * Gamm)
        dblmatrix[6,n] = fac1i
        dblmatrix[7,n] = fac2i
        dblmatrix[8,n] = fac2i - ek + ekq
        n = n + 1

    sdr00 = numba.cuda.shared.array((N,N), dtype=numba.types.complex128)
    sdr42 = numba.cuda.shared.array((N,N), dtype=numba.types.complex128)
    sdr11 = numba.cuda.shared.array((N,N), dtype=numba.types.complex128)
    sdr20 = numba.cuda.shared.array((N,N), dtype=numba.types.complex128)
    sdr53 = numba.cuda.shared.array((N,N), dtype=numba.types.complex128)
    sdsdr3173 = numba.cuda.shared.array((N,N), dtype=numba.types.complex128)
    sdr62 = numba.cuda.shared.array((N,N), dtype=numba.types.complex128)

    for s in range(0, N):
        for dummy in range(0, N):
            both = s + dummy
            sdr00[s,dummy] = singmatrix[0,dummy] - dblmatrix[0,both]
            sdr42[s,dummy] = singmatrix[4,dummy] + dblmatrix[2,both]
            sdr11[s,dummy] = -singmatrix[1,dummy] + dblmatrix[1,both]
            sdr20[s,dummy] = singmatrix[2,dummy] - dblmatrix[0,both]
            sdr53[s,dummy] = singmatrix[5,dummy] + dblmatrix[3,both]
            sdsdr3173[s,dummy] = -singmatrix[3,dummy] + dblmatrix[1,both] - singmatrix[7,dummy] + dblmatrix[3,both]
            sdr62[s,dummy] = singmatrix[6,dummy] + dblmatrix[2,both]


    for n in range(0, N):
        nmod = n + N - 1
        for alpha in range(0, N):

            for beta in range(0, N):
                abdiff = alpha - beta + N - 1

                d1part = -2 * complex(0, 1) * dblmatrix[8,abdiff]
                d2part = -2 * complex(0, 1) * dblmatrix[6,abdiff]

                for gamma in range(0, N):
                    bgdiff = beta - gamma + N - 1
                    agdiff = alpha - gamma + N - 1

                    d1 = d1part * dblmatrix[6,bgdiff] * dblmatrix[7,agdiff]
                    d2 = d2part * dblmatrix[7,agdiff] * dblmatrix[8,bgdiff]

                    besspart1 = dblmatrix[5,gamma - nmod] * dblmatrix[4,alpha - nmod]


                    for s in range(0, N):
                        smod = s + N - 1
                        tau = s + beta
                        

                        p1p = dblmatrix[6,bgdiff] * (sdr00[s,alpha] - sdr42[s,alpha])
                        p2p = dblmatrix[7,agdiff] * (sdr00[s,beta] + sdr42[s,beta])
                        p3p = dblmatrix[8,abdiff] * (sdr11[s,gamma] - sdr53[s,gamma])

                        p1m = dblmatrix[6,bgdiff] * (sdr20[s,alpha] - sdr62[s,alpha])
                        p2m = dblmatrix[7,agdiff] * (sdr20[s,beta] + singmatrix[6,beta] - dblmatrix[2,tau])
                        p3m = dblmatrix[8,abdiff] * (sdsdr3173[s,gamma])

                        pp1p = dblmatrix[6,abdiff] * (sdr11[s,gamma] - sdr53[s,gamma])
                        pp2p = dblmatrix[7,agdiff] * (sdr11[s,beta] + singmatrix[5,beta] - dblmatrix[3,tau])
                        pp3p = dblmatrix[8,bgdiff] * (sdr00[s,alpha] - sdr42[s,alpha])

                        pp1m = dblmatrix[6,abdiff] * (sdsdr3173[s,gamma])
                        pp2m = dblmatrix[7,agdiff] * (-singmatrix[3,beta] + dblmatrix[1,tau] + singmatrix[7,beta] - dblmatrix[3,tau])
                        pp3m = dblmatrix[8,bgdiff] * (sdr20[s,alpha] - sdr62[s,alpha])

                        omint1p = singmatrix[8,s] * ((p1p + p2p + p3p) / d1)
                        omint1m = singmatrix[9,s] * ((p1m + p2m + p3m) / d1)

                        omint2p = singmatrix[8,s] * ((pp1p + pp2p + pp3p) / d2)
                        omint2m = singmatrix[9,s] * ((pp1m + pp2m + pp3m) / d2)

                        bess1part2 = dblmatrix[4,beta - smod] * dblmatrix[4,alpha - smod] * besspart1
                        bess2part2 = dblmatrix[5,gamma - smod] * dblmatrix[5,beta - smod] * besspart1


                        for l in range(0, N):
                            lmod = l + N - 1

                            bess1 = bess1part2 * dblmatrix[5,gamma - lmod] * dblmatrix[4,beta - lmod] 

                            grgl = bess1 * (omint1p - omint1m)

                            bess2 = bess2part2 * dblmatrix[5,beta - lmod] * dblmatrix[4,alpha - lmod]

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

# Setting up first 10 points on 100 point interval [0.001, pi/4]
qinitial = 0.001
qfinal = .0785
spacing = 10

# SET PARAMETERS AND LIMITS OF INTEGRATION
kxi = - math.pi / a
kxf = math.pi / a

kyi = - math.pi / a
kyf = math.pi / a

resultArr = np.zeros(spacing)
errorArr = np.zeros(spacing)
timeArr = np.zeros(spacing)

j = 0
for i in np.linspace(qinitial, qfinal, spacing):

    helpers.setqx(i)
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
for i in np.linspace(qinitial, qfinal, spacing):
    print('%5.3f, %11.8E, %5.3E, %5.3E' % (i, resultArr[j], errorArr[j], timeArr[j]))
    j = j + 1

toc = time.time()
print('================================================================\n')
print('Process completed successfully!')
print('Total time is ', toc-tic, 'seconds.')