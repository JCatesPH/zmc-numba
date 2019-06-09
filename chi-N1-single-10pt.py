
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
hOmg = 0.5  # Photon energy eV
a = 4  # AA
A = 4  # hbar^2/(2m)=4 evAA^2 (for free electron mass)
rati = 0.01  # ratio
eE0 = rati * ((hOmg) ** 2) / (2 * np.sqrt(A * mu))
# print(eE0)
Gamm = 0.003  # Gamma in eV.
KT = 1 * 10 ** (-6)
shift = A * (eE0 / hOmg) ** 2

Gammsq = Gamm ** 2

@numba.cuda.jit(device=True)        
def modDsN2(x):
    N = 1
    dds = 0
    # ds = 0 # UNUSED
    qx = cudahelpers.getqx()
    ek = A * (math.sqrt((x[0]) ** 2 + (x[1]) ** 2)) ** 2 + A * (eE0 / hOmg) ** 2
    ekq = A * (math.sqrt((x[0] + qx) ** 2 + (x[1] + 0) ** 2)) ** 2 + A * (eE0 / hOmg) ** 2
    xk = 2 * A * eE0 * math.sqrt((x[0]) ** 2 + (x[1]) ** 2) / hOmg ** 2
    xkq = 2 * A * eE0 * math.sqrt((x[0] + qx) ** 2 + (x[1] + 0) ** 2) / hOmg ** 2

    singmatrix = numba.cuda.shared.array((10,N),dtype=numba.types.complex64)  

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

        singmatrix[8,n] = cudahelpers.my_heaviside(mu - chinu)
        singmatrix[9,n] = cudahelpers.my_heaviside(mu + chinu)
        i = i + 1
        n = n + 1

    numba.cuda.syncthreads()

    size_dbl = 5
    dblmatrix = numba.cuda.shared.array((9,size_dbl),dtype=numba.types.complex64)

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

        dblmatrix[4,n] = cudahelpers.besselj(i, xk)
        dblmatrix[5,n] = cudahelpers.besselj(i, xkq)

        fac1i = ek - ekq + xi
        fac2i = complex(fac1i, 2 * Gamm)
        dblmatrix[6,n] = fac1i
        dblmatrix[7,n] = fac2i
        dblmatrix[8,n] = fac2i - ek + ekq
        n = n + 1

    numba.cuda.syncthreads()

    for n in range(0, N):
        for alpha in range(0, N):
            for beta in range(0, N):
                for gamma in range(0, N):
                    for s in range(0, N):
                        for l in range(0, N):
                            p1p = dblmatrix[6,beta - gamma + N - 1] * (singmatrix[0,alpha] - dblmatrix[0,s + alpha] - singmatrix[4,alpha] + dblmatrix[2,s + alpha])
                            p2p = dblmatrix[7,alpha - gamma + N - 1] * (singmatrix[0,beta] - dblmatrix[0,s + beta] + singmatrix[4,beta] - dblmatrix[2,s + beta])
                            p3p = dblmatrix[8,alpha - beta + N - 1] * (-singmatrix[1,gamma] + dblmatrix[1,s + gamma] - singmatrix[5,gamma] + dblmatrix[3,s + gamma])

                            p1m = dblmatrix[6,beta - gamma + N - 1] * (singmatrix[2,alpha] - dblmatrix[0,s + alpha] - singmatrix[6,alpha] + dblmatrix[2,s + alpha])

                            p2m = dblmatrix[7,alpha - gamma + N - 1] * ( singmatrix[2,beta] - dblmatrix[0,s + beta] + singmatrix[6,beta] - dblmatrix[2,s + beta])

                            p3m = dblmatrix[8,alpha - beta + N - 1] * (-singmatrix[3,gamma] + dblmatrix[1,s + gamma] - singmatrix[7,gamma] + dblmatrix[3,s + gamma])

                            d1 = -2 * complex(0, 1) * dblmatrix[6,beta - gamma + N - 1] * dblmatrix[7,alpha - gamma + N - 1] * dblmatrix[8,alpha - beta + N - 1]

                            omint1p = singmatrix[8,s] * ((p1p + p2p + p3p) / d1)

                            omint1m = singmatrix[9,s] * ((p1m + p2m + p3m) / d1)

                            bess1 = dblmatrix[5,gamma - n + N - 1] * dblmatrix[5,gamma - l + N - 1] * dblmatrix[4,beta - l + N - 1] * dblmatrix[4,beta - s + N - 1] * dblmatrix[4,alpha - s + N - 1] * dblmatrix[4,alpha - n + N - 1]

                            grgl = bess1 * (omint1p - omint1m)

                            pp1p = dblmatrix[6,alpha - beta + N - 1] * (-singmatrix[1,gamma] + dblmatrix[1,s + gamma] - singmatrix[5,gamma] + dblmatrix[3,s + gamma])

                            pp2p = dblmatrix[7,alpha - gamma + N - 1] * (-singmatrix[1,beta] + dblmatrix[1,s + beta] + singmatrix[5,beta] - dblmatrix[3,s + beta])

                            pp3p = dblmatrix[8,beta - gamma + N - 1] * (singmatrix[0,alpha] - dblmatrix[0,s + alpha] - singmatrix[4,alpha] + dblmatrix[2,s + alpha])

                            pp1m = dblmatrix[6,alpha - beta + N - 1] * (-singmatrix[3,gamma] + dblmatrix[1,s + gamma] - singmatrix[7,gamma] + dblmatrix[3,s + gamma])

                            pp2m = dblmatrix[7,alpha - gamma + N - 1] * (-singmatrix[3,beta] + dblmatrix[1,s + beta] + singmatrix[7,beta] - dblmatrix[3,s + beta])

                            pp3m = dblmatrix[8,beta - gamma + N - 1] * (singmatrix[2,alpha] - dblmatrix[0,s + alpha] - singmatrix[6,alpha] + dblmatrix[2,s + alpha])

                            d2 = -2 * complex(0, 1) * dblmatrix[6,alpha - beta + N - 1] * dblmatrix[7,alpha - gamma + N - 1] * dblmatrix[8,beta - gamma + N - 1]

                            omint2p = singmatrix[8,s] * ((pp1p + pp2p + pp3p) / d2)

                            omint2m = singmatrix[9,s] * ((pp1m + pp2m + pp3m) / d2)

                            bess2 = dblmatrix[5,gamma - n + N - 1] * dblmatrix[5,gamma - s + N - 1] * dblmatrix[5,beta - s + N - 1] * dblmatrix[5,beta - l + N - 1] * dblmatrix[4,alpha - l + N - 1] * dblmatrix[4,alpha - n + N - 1]

                            glga = bess2 * (omint2p - omint2m)

                            dds = dds + Gamm * (grgl + glga)
    return -4 * dds.real / math.pi ** 2



tic = time.time()

print('Following values are constant for all integrations.')
print('\n========================================================')
print('\ndepth = 2')
print('sigma_multiplication = 1,000,000')
print('num_trials = 5')
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

resultArr = np.zeros(15)
errorArr = np.zeros(15)
timeArr = np.zeros(15)
j = 0
for i in np.linspace(.01, .315, 15):

    cudahelpers.setqx(i)
    MC = ZMCIntegral.MCintegral(modDsN2,[[kxi,kxf],[kyi,kyf]])
    # Setting the zmcintegral parameters
    MC.depth = 2
    MC.sigma_multiplication = 1E6
    MC.num_trials = 5
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
for i in np.linspace(.01, .315, 15):
    print('%5.3f, %11.8E, %5.3E, %5.3E' % (i, resultArr[j], errorArr[j], timeArr[j]))
    j = j + 1

toc = time.time()
print('================================================================\n')
print('Process completed successfully!')
print('Total time is ', toc-tic, 'seconds.')
