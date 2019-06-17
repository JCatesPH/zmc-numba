import datetime
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


sq3 = math.sqrt(float(3))
sq3pi = sq3 * math.pi
pi2 = 2 * math.pi

qx = 0.1 / pi2
qy = 0

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


    om = getom()

    hkxky = exi(x[0] * pi2, 1) * (
                exi((x[0] * math.pi - sq3pi * x[1]), 1) + exi(
            (x[0] * math.pi + sq3pi * x[1]), 1) + exi((x[0] * pi2), -1))
    chkxky = exi(x[0] * pi2, -1) * (
                exi((x[0] * math.pi - sq3pi * x[1]), -1) + exi(
            (x[0] * math.pi + sq3pi * x[1]), -1) + exi((x[0] * pi2), 1))
    hkxqxky = exi((x[0] + qx) * pi2, 1) * (
                exi(((x[0] + qx) * math.pi - sq3pi * (x[1] + qy)), 1) + exi(
            ((x[0] + qx) * math.pi + sq3pi * (x[1] + qy)), 1) + exi(
            ((x[0] + qx) * pi2), -1))
    chkxqxky = exi((x[0] + qx) * pi2, -1) * (
                exi(((x[0] + qx) * math.pi - sq3pi * (x[1] + qy)), -1) + exi(
            ((x[0] + qx) * math.pi + sq3pi * (x[1] + qy)), -1) + exi(
            ((x[0] + qx) * pi2), 1))

    ekp = math.sqrt(
        3 + 2 * math.cos(sq3pi * x[1] * 2) + 4 * math.cos(sq3pi * x[1]) * math.cos(
            3 * x[0] * math.pi))
    ekm = -ekp
    ekqp = math.sqrt(3 + 2 * math.cos(sq3 * (x[1] + qy) * pi2) + 4 * math.cos(
        sq3pi * (x[1] + qy)) * math.cos(3 * (x[0] + qx) * math.pi))
    ekqm = -ekqp

    hp = (chkxky * hkxqxky / (ekp * ekqp) + 1) * (hkxky * chkxqxky / (ekp * ekqp) + 1) / 4
    hm = (chkxky * hkxqxky / (ekp * ekqp) - 1) * (hkxky * chkxqxky / (ekp * ekqp) - 1) / 4

    a1 = hp * (fermi(ekm) - fermi(ekqm)) / (ekm - ekqm + 1j * eta + om)
    a2 = hm * (fermi(ekm) - fermi(ekqp)) / (ekm - ekqp + 1j * eta + om)
    a3 = hm * (fermi(ekp) - fermi(ekqm)) / (ekp - ekqm + 1j * eta + om)
    a4 = hp * (fermi(ekp) - fermi(ekqp)) / (ekp - ekqp + 1j * eta + om)

    a = -2 * (a1 + a2 + a3 + a4)

    return a.imag


###################################################################
# Setting parameters and printing them

depths = 3
sigmults = 100
trials = 10

print('\n========================================================\n')

print(__file__)
print(datetime.date.today())

print('\ndepth = ', depths)
print('sigma_multiplication = ', sigmults)
print('num_trials = ', trials)

kxi = -1 / 3
kxf = 1 / 3

kyi = -0.1925
kyf = 0.385

print('\nkxi = ', kxi)
print('kxf = ', kxf)

print('\nkyi = ', kyi)
print('kyf = ', kyf)

beg = 1
end = 51
spacing = 50

print('\nlinspace parameters: %3.3f,%3.3f,%3.3f' % (beg, end, spacing))

print('=======================================================')
omArr = np.zeros(spacing)
resultArr = np.zeros(spacing)
errorArr = np.zeros(spacing)
timeArr = np.zeros(spacing)

tic = time.time()
j = 0
for i in np.linspace(beg, end, spacing):
    omArr[j] = 1.98 + i * 0.05 / spacing
    setom(omArr[j])
    MC = ZMCIntegral.MCintegral(integrand, [[kxi, kxf], [kyi, kyf]])
    # Setting the zmcintegral parameters
    MC.depth = depths
    MC.sigma_multiplication = sigmults
    MC.num_trials = trials
    start = time.time()
    result = MC.evaluate()
    print('================================================================')
    print('Result for om = ', omArr[j], ': ', result[0], ' with error: ', result[1])
    print('================================================================')
    end = time.time()
    print('Computed in ', end - start, ' seconds.')
    print('================================================================')
    resultArr[j] = result[0]
    errorArr[j] = result[1]
    timeArr[j] = end - start
    j = j + 1

print('\n\n================================================================')


print('All values in csv format:')
print('om,Integral,Error,Time')
for j in range(0, spacing):
    print('%5.3f, %11.8E, %5.3E, %5.3E' % (omArr[j], resultArr[j], errorArr[j], timeArr[j]))


toc = time.time()
print('================================================================\n')
print('Process completed successfully!')
print('Total time is ', toc - tic, 'seconds.')
