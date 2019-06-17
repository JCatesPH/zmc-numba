import math
import time
import numpy as np
from numba import cuda
import ZMCIntegral

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
def exi(x, sgn):
    return math.cos(x) + complex(0, 1) * sgn * math.sin(x)


@cuda.jit(device=True)
def integrand(x):
    # Note: q=2pi/a
    # Inset of fig 2  (phi=0)

    # 2  the Kpoint and M-point: reference of testing

    # qy=[]

    # In PRB 
    qx = 0.1 / (2 * math.pi)
    qy = 0
    sq3 = math.sqrt(float(3))

    sq3x1 = sq3 * x[1]
    sq3x1pi = sq3x1 * math.pi
    phi = x[0] * 2 * math.pi
    tau = sq3x1 * 2 * math.pi / 2
    rho = phi / 2
    psi = x[0] + qx
    eta = psi * 2 * math.pi
    zeta = x[1] + qy
    sq3zetapi = sq3 * math.pi * zeta

    nu = eta / 2 - sq3 * (zeta) * 2 * math.pi / 2
    mu = eta / 2 + sq3 * (zeta) * 2 * math.pi / 2


    om = getom()

    hkxky = exi(phi, 1) * (exi((rho - tau), 1) + exi((rho + tau), 1) + exi(phi, -1))
    chkxky = exi(phi, -1) * (exi((rho - tau), -1) + exi((rho + tau), -1) + exi(phi, 1))

    hkxqxky = exi(eta, 1) * (exi(nu, 1) + exi(mu, 1) + exi(eta, -1))
    chkxqxky = exi(eta, -1) * (exi(nu, -1) + exi(mu, -1) + exi(eta, 1))

    ekp = math.sqrt(3 + 2 * math.cos(sq3x1pi * 2) + 4 * math.cos(sq3x1pi) * math.cos(3 * x[0] * math.pi))
    ekm = -ekp

    ekqp = math.sqrt(3 + 2 * math.cos(sq3zetapi * 2) + 4 * math.cos(sq3zetapi) * math.cos(3 * psi * math.pi))
    ekqm = -ekqp

    hp = (chkxky * hkxqxky / (ekp * ekqp) + 1) * (hkxky * chkxqxky / (ekp * ekqp) + 1) / 4
    hm = (chkxky * hkxqxky / (ekp * ekqp) - 1) * (hkxky * chkxqxky / (ekp * ekqp) - 1) / 4

    a1 = hp * (fermi(ekm) - fermi(ekqm)) / (ekm - ekqm + 1j * eta + om)
    a2 = hm * (fermi(ekm) - fermi(ekqp)) / (ekm - ekqp + 1j * eta + om)
    a3 = hm * (fermi(ekp) - fermi(ekqm)) / (ekp - ekqm + 1j * eta + om)
    a4 = hp * (fermi(ekp) - fermi(ekqp)) / (ekp - ekqp + 1j * eta + om)

    a = (a1 + a2 + a3 + a4) / (2 * math.pi) ** 2

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
