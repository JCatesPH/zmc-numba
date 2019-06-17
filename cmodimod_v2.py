import math
import cmath
import time
import numpy as np
from numba import cuda
import ZMCIntegral

# import time
# start = time.time()

cef = 0. + 0j
charge = 1.600000000 * 10 ** -19
lattice = 1.420000000 * 10 ** (-10)
t = 4.8000000 * 10 ** (-19)
eta = 1.0000000 * 10 ** (-3)


def setom(ome):
    global om
    om = ome
    return


@cuda.jit(device=True)
def getom():
    return om


@cuda.jit(device=True)
def fermi(x):
    if (x.real > cef.real) and (x.imag > cef.imag):
        return 0
    else:
        return 1


@cuda.jit(device=True)
def integrand(x):
    qx = 0.1 / (2 * math.pi)
    qy = 0
    om = getom()

    sq3 = cmath.sqrt(3)
    rho = x[0] * 2 * math.pi
    rho2 = rho / 2 
    irho = complex(0,rho)
    tau = sq3 * x[1] * 2 * math.pi
    tau2 = tau / 2

    irt2 = complex(0,rho2 + tau2)
    irt2d = complex(0,rho2 - tau2)
    mu = x[0] + qx
    mu2pi = mu * 2 * math.pi
    nu = sq3 * (x[1] + qy) * 2 * math.pi

    # MORE TO COME

    hkxky = cmath.exp(irho) * (cmath.exp(irt2d) + cmath.exp(irt2) + cmath.exp(-irho))
    chkxky = cmath.exp(-irho) * (cmath.exp(-irt2d) + cmath.exp(-irt2) + cmath.exp(irho))
    hkxqxky = cmath.exp(complex(0, 1) * mu2pi) * (cmath.exp(complex(0, 1) * (mu2pi / 2 - nu / 2)) + cmath.exp(complex(0, 1) * (mu2pi / 2 + nu / 2)) + cmath.exp(-complex(0, mu2pi)))
    chkxqxky = cmath.exp(-complex(0, 1) * mu2pi) * (cmath.exp(-complex(0, 1) * (mu2pi / 2 - nu / 2)) + cmath.exp(-complex(0, 1) * (mu2pi / 2 + nu / 2)) + cmath.exp(complex(0, mu2pi)))

    ekp = cmath.sqrt(3 + 2 * cmath.cos(tau) + 4 * cmath.cos( sq3 * x[1] * cmath.pi) * cmath.cos(3 * x[0] * cmath.pi))
    ekm = -ekp
    ekqp = cmath.sqrt(3 + 2 * cmath.cos(nu) + 4 * cmath.cos(sq3 * (x[1] + qy) * cmath.pi) * cmath.cos(3 * (mu) * cmath.pi))
    ekqm = -ekqp

    hp = (chkxky * hkxqxky / (ekp * ekqp) + 1) * (hkxky * chkxqxky / (ekp * ekqp) + 1) / 4
    hm = (chkxky * hkxqxky / (ekp * ekqp) - 1) * (hkxky * chkxqxky / (ekp * ekqp) - 1) / 4

    a1 = hp * (fermi(ekm) - fermi(ekqm)) / (ekm - ekqm + 1j * eta + om)
    a2 = hm * (fermi(ekm) - fermi(ekqp)) / (ekm - ekqp + 1j * eta + om)
    a3 = hm * (fermi(ekp) - fermi(ekqm)) / (ekp - ekqm + 1j * eta + om)
    a4 = hp * (fermi(ekp) - fermi(ekqp)) / (ekp - ekqp + 1j * eta + om)

    a = (a1 + a2 + a3 + a4) / (2 * cmath.pi) ** 2

    return a.imag


##############################################################################
# # Defining calculation parameters and printing them
kxi = -1 / 3
kxf = 1 / 3

kyi = -0.1925
kyf = 0.385

beg = 1
end = 51
spacing = 50

depths = 3
sigmults = 100
trials = 10

print('Following values are constant for all integrations.')
print('\n========================================================')

print('\ndepth = ', depths)
print('sigma_multiplication = ', sigmults)
print('num_trials = ', trials)

print('\nkxi = ', kxi)
print('kxf = ', kxf)
print('kyi = ', kyi)
print('kyf = ', kyf)

print('\nLinspace parameters : %3f, %3f, %3f' % (beg, end, spacing))
print('========================================================')

tic = time.time()

resultArr = np.zeros(spacing)
errorArr = np.zeros(spacing)
timeArr = np.zeros(spacing)

##############################################################################


j = 0
for i in np.linspace(beg, end, spacing):
    setom(1.98 + i * 0.05 / (50))
    MC = ZMCIntegral.MCintegral(integrand, [[kxi, kxf], [kyi, kyf]])
    # Setting the zmcintegral parameters
    MC.depth = depths
    MC.sigma_multiplication = sigmults
    MC.num_trials = trials
    start = time.time()
    result = MC.evaluate()
    print('================================================================')
    print('Result for r = ', i, ': ', result[0], ' with error: ', result[1])
    print('================================================================')
    end = time.time()
    print('Computed in ', end - start, ' seconds.')
    print('================================================================')
    resultArr[j] = result[0]
    errorArr[j] = result[1]
    timeArr[j] = end - start
    j = j + 1

print('================================================================')

j = 0
print('All values in csv format:')
print('r,Integral,Error,Time')
for i in np.linspace(beg, end, spacing):
    print('%5.3f, %11.8E, %5.3E, %5.3E' % (i, resultArr[j], errorArr[j], timeArr[j]))
    j = j + 1

toc = time.time()
print('================================================================\n')
print('Process completed successfully!')
print('Total time is ', toc - tic, 'seconds.')
