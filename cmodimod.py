import cmath
import time
import numpy as np
from numba import cuda
import ZMCIntegral

# import time
# start = time.time()

ef = 0.
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
    if x > ef:
        return 0
    else:
        return 1


@cuda.jit(device=True)
def integrand(x):
    qx = 0.1 / (2 * cmath.pi)
    qy = 0
    om = getom()

    hkxky = cmath.exp(complex(0, 1) * x[0] * 2 * cmath.pi) * (cmath.exp(
        complex(0, 1) * (x[0] * 2 * cmath.pi / 2 - cmath.sqrt(3) * x[1] * 2 * cmath.pi / 2)) + cmath.exp(
        complex(0, 1) * (x[0] * 2 * cmath.pi / 2 + cmath.sqrt(3) * x[1] * 2 * cmath.pi / 2)) + cmath.exp(
        -complex(0, 1)(x[0] * 2 * cmath.pi)))
    chkxky = cmath.exp(-complex(0, 1) * x[0] * 2 * cmath.pi) * (cmath.exp(
        -complex(0, 1) * (x[0] * 2 * cmath.pi / 2 - cmath.sqrt(3) * x[1] * 2 * cmath.pi / 2)) + cmath.exp(
        -complex(0, 1) * (x[0] * 2 * cmath.pi / 2 + cmath.sqrt(3) * x[1] * 2 * cmath.pi / 2)) + cmath.exp(
        complex(0, 1) * (x[0] * 2 * cmath.pi)))
    hkxqxky = cmath.exp(complex(0, 1) * (x[0] + qx) * 2 * cmath.pi) * (cmath.exp(
        complex(0, 1) * ((x[0] + qx) * 2 * cmath.pi / 2 - cmath.sqrt(3) * (x[1] + qy) * 2 * cmath.pi / 2)) + cmath.exp(
        complex(0, 1) * ((x[0] + qx) * 2 * cmath.pi / 2 + cmath.sqrt(3) * (x[1] + qy) * 2 * cmath.pi / 2)) + cmath.exp(
        -complex(0, 1) * ((x[0] + qx) * 2 * cmath.pi)))
    chkxqxky = cmath.exp(-complex(0, 1) * (x[0] + qx) * 2 * cmath.pi) * (cmath.exp(
        -complex(0, 1) * ((x[0] + qx) * 2 * cmath.pi / 2 - cmath.sqrt(3) * (x[1] + qy) * 2 * cmath.pi / 2)) + cmath.exp(
        -complex(0, 1) * ((x[0] + qx) * 2 * cmath.pi / 2 + cmath.sqrt(3) * (x[1] + qy) * 2 * cmath.pi / 2)) + cmath.exp(
        complex(0, 1) * ((x[0] + qx) * 2 * cmath.pi)))

    ekp = cmath.sqrt(3 + 2 * cmath.cos(cmath.sqrt(3) * x[1] * 2 * cmath.pi) + 4 * cmath.cos(
        cmath.sqrt(3) * x[1] * cmath.pi) * cmath.cos(3 * x[0] * cmath.pi))
    ekm = -ekp
    ekqp = cmath.sqrt(3 + 2 * cmath.cos(cmath.sqrt(3) * (x[1] + qy) * 2 * cmath.pi) + 4 * cmath.cos(
        cmath.sqrt(3) * (x[1] + qy) * cmath.pi) * cmath.cos(3 * (x[0] + qx) * cmath.pi))
    ekqm = -ekqp

    hp = (chkxky * hkxqxky / (ekp * ekqp) + 1) * (hkxky * chkxqxky / (ekp * ekqp) + 1) / 4
    hm = (chkxky * hkxqxky / (ekp * ekqp) - 1) * (hkxky * chkxqxky / (ekp * ekqp) - 1) / 4

    a1 = hp * (fermi(ekm) - fermi(ekqm)) / (ekm - ekqm + 1j * eta + om)
    a2 = hm * (fermi(ekm) - fermi(ekqp)) / (ekm - ekqp + 1j * eta + om)
    a3 = hm * (fermi(ekp) - fermi(ekqm)) / (ekp - ekqm + 1j * eta + om)
    a4 = hp * (fermi(ekp) - fermi(ekqp)) / (ekp - ekqp + 1j * eta + om)

    a = (a1 + a2 + a3 + a4) / (2 * cmath.pi) ** 2

    return a.imag


tic = time.time()

print('Following values are constant for all integrations.')
print('\n========================================================')
print('\ndepth = 3')
print('sigma_multiplication = 100')
print('num_trials = 10')

kxi = -1 / 3
kxf = 1 / 3

kyi = -0.1925
kyf = 0.385

resultArr = np.zeros(100)
errorArr = np.zeros(100)
timeArr = np.zeros(100)
j = 0
for i in range(1, 51, 1):
    setom(1.98 + i * 0.05 / (50))
    MC = ZMCIntegral.MCintegral(integrand, [[kxi, kxf], [kyi, kyf]])
    # Setting the zmcintegral parameters
    MC.depth = 3
    MC.sigma_multiplication = 10000
    MC.num_trials = 10
    start = time.time()
    result = MC.evaluate()
    print('Result for om = ', 1.98 + i * 0.05 / (50), ': ', result[0], ' with error: ', result[1])
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
for i in range(1, 51, 1):
    print('%5.3f, %11.8E, %5.3E, %5.3E' % (i, resultArr[j], errorArr[j], timeArr[j]))
    j = j + 1

toc = time.time()
print('================================================================\n')
print('Process completed successfully!')
print('Total time is ', toc - tic, 'seconds.')