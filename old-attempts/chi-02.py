#!/home/jmcates/miniconda3/envs/zmcint/bin/python
# coding: utf-8

# # # Attempt to evaluate the integral of the function given by Dr. Tse using ZMCintegral
# #  The information given is here:
# Mahmoud:
# "The integrand is Ds(kx,ky,qx,qy)/(2*pi)^3, and the limits of integration are kx=[-pi/a,pi/a],ky=[-pi/a,pi/a] , qx=[-pi/a,pi/a] and qy=[-pi/a,pi/a]."
# "For qx and qy it is more efficient to use qx=[0.001,pi/a] and qy=0, because of the symmetry of the problem. kx and ky should be as we said before kx=[-pi/a,pi/a],ky=[-pi/a,pi/a]."
#
# Dr. Tse:
# "Hi Jalen, what we need is a plot of the integrated result as a function of qx. My postdoc Mahmoud has a plot for that he obtained previously from another integration method that we can compare your MC results with. "


# # # CHANGE LOG :
#   v4 :
#       Extended Bessel function to eight terms and applied Horner's Algorithm to it for numerical efficiency.
#           Eight terms is not enough as z > 1. So, ELEVEN terms will be used. n = 11 => err <~ E-14
#       Removing commented out code for readability. Retaining copy with commented code as well.


# The import statements
import math
import time
import numpy as np
from numba import cuda
import scipy
import scipy.special
import ZMCIntegral
import helpers

# Define constants in function

mu = 0.1  # Fermi-level
hOmg = 0.5  # Photon energy eV
a = 4  # AA
A = 4  # hbar^2/(2m)=4 evAA^2 (for free electron mass)
rati = 0.01  # ratio
eE0 = rati * ((hOmg) ** 2) / (2 * np.sqrt(A * mu))
# print(eE0)
Gamm = 0.003  # Gamma in eV.
shift = A * (eE0 / hOmg) ** 2


# Function given with slight modification. I replaced all calls to kx, ky, qx, and qy with x[0], x[1], x[2], and x[3] respectively. This modification effectively "vectorizes" the input.

# Initialize the variable qx as global
qx = 0
# print('qx = ', qx)
# Helper function to set qx
def setqx(qxi):
	global qx
	qx = qxi
	return

# Helper function to get qx
@cuda.jit(device=True)
def getqx():
    return qx

@cuda.jit(device=True)
def modDs_real(x):
    dds = 0
    qx = getqx()


    ek = A * (math.sqrt((x[0]) ** 2 + (x[1]) ** 2)) ** 2 + A * (eE0 / hOmg) ** 2
    ekq = A * (math.sqrt((x[0] + qx) ** 2 + (x[1] + 0) ** 2)) ** 2 + A * (eE0 / hOmg) ** 2
    xk = 2 * A * eE0 * math.sqrt((x[0]) ** 2 + (x[1]) ** 2) / hOmg ** 2
    xkq = 2 * A * eE0 * math.sqrt((x[0] + qx) ** 2 + (x[1] + 0) ** 2) / hOmg ** 2

    ts1 = ek - hOmg / 2
    ts2 = ekq - hOmg / 2
    ts3 = ek + hOmg / 2
    ts4 = ekq + hOmg / 2

    arc2ts1 = math.atan2(Gamm, ts1)
    arc2ts2 = math.atan2(Gamm, ts2)
    arc2ts3 = math.atan2(Gamm, ts3)
    arc2ts4 = math.atan2(Gamm, ts4)

    taninv1kp = 2 * arc2ts1
    taninv1kqp = 2 * arc2ts2
    taninv1km = 2 * arc2ts3
    taninv1kqm = 2 * arc2ts4

    squared1 = ek - hOmg/2
    squared2 = ekq - hOmg/2
    squared3 = ek + hOmg/2
    squared4 = ekq + hOmg/2

    logged1 = Gamm**2 + squared1**2
    logged2 = Gamm**2 + squared2**2
    logged3 = Gamm**2 + squared3**2
    logged4 = Gamm**2 + squared4**2

    logged1 = Gamm**2 + (ek - hOmg/2)**2
    logged2 = Gamm**2 + (ekq - hOmg/2)**2
    logged3 = Gamm**2 + (ek + hOmg/2)**2
    logged4 = Gamm**2 + (ekq + hOmg/2)**2

    ln1 = math.log(logged1)
    ln2 = math.log(logged2)
    ln3 = math.log(logged3)
    ln4 = math.log(logged4)

    lg1kp = complex(0, 1) * ln1
    lg1kqp = complex(0, 1) * ln2
    lg1km = complex(0, 1) * ln3
    lg1kqm = complex(0, 1) * ln4

    heavi1 = mu - hOmg / 2
    heavi2 = mu + hOmg / 2

    ferp = helpers.my_heaviside(heavi1)
    ferm = helpers.my_heaviside(heavi2)

    taninv2k = 2 * math.atan2(Gamm, ek - mu)
    taninv2kq = 2 * math.atan2(Gamm, ekq - mu)

    lg2k = complex(0, 1) * math.log(Gamm ** 2 + (ek - mu) ** 2)
    lg2kq = complex(0, 1) * math.log(Gamm ** 2 + (ekq - mu) ** 2)

    besk = helpers.my_Bessel(xk)
    beskq = helpers.my_Bessel(xkq)

    fac1 = ek - ekq

    fac2 = fac1 + 2 * complex(0, 1) * Gamm
    fac3 = fac2 - ek + ekq


    # NOTE: N = 1 implies all loops below will be evaluated once.
    p1p = fac1 * (taninv1kp - taninv2k - lg1kp + lg2k)
    p2p = fac2 * (taninv1kp - taninv2k + lg1kp - lg2k)
    p3p = fac3 * (-taninv1kqp + taninv2kq - lg1kqp + lg2kq)

    p1m = fac1 * (taninv1km - taninv2k - lg1km + lg2k)
    p2m = fac2 * (taninv1km - taninv2k + lg1km - lg2k)
    p3m = fac3 * (-taninv1kqm + taninv2kq - lg1kqm + lg2kq)

    d1 = -2 * complex(0, 1) * fac1 * fac2 * fac3

    omint1p = ferp * ((p1p + p2p + p3p) / d1)
    omint1m = ferm * ((p1m + p2m + p3m) / d1)

    bess1 = beskq * beskq * besk * besk * besk * besk

    grgl = bess1 * (omint1p - omint1m)

    pp1p = fac1 * (-taninv1kqp + taninv2kq - lg1kqp + lg2kq)
    pp2p = fac2 * (-taninv1kqp + taninv2kq + lg1kqp - lg2kq)
    pp3p = fac3 * (taninv1kp - taninv2k - lg1kp + lg2k)

    pp1m = fac1 * (-taninv1kqm + taninv2kq - lg1kqm + lg2kq)
    pp2m = fac2 * (-taninv1kqm + taninv2kq + lg1kqm - lg2kq)

    pp3m = fac3 * (taninv1km - taninv2k - lg1km + lg2k)

    d2 = -2 * complex(0, 1) * fac1 * fac2 * fac3

    omint2p = ferp * ((pp1p + pp2p + pp3p) / d2)
    omint2m = ferm * ((pp1m + pp2m + pp3m) / d2)

    bess2 = beskq * beskq * beskq * beskq * besk * besk

    glga = bess2 * (omint2p - omint2m)

    dds = dds + Gamm * (grgl + glga)

    return - 4 * dds.real / math.pi**2


# # Mahmoud:
#  "The integrand is Ds(kx,ky,qx,qy)/(2*pi)^3, and the limits of integration are kx=[-pi/a,pi/a],ky=[-pi/a,pi/a] , qx=[-pi/a,pi/a] and qy=[-pi/a,pi/a]."
#
# "For qx and qy it is more efficient to use qx=[0.001,pi/a] and qy=0, because of the symmetry of the problem. kx and ky should be as we said before kx=[-pi/a,pi/a],ky=[-pi/a,pi/a]."


# Introducing suggested values of integration.


tic = time.time()

# MC.available_GPU=[0]

print('Following values are constant for all integrations.')
print('\n========================================================')
print('\ndepth = 3')
print('sigma_multiplication = 100')
print('num_trials = 10')
print('available_GPU = [0]')
print('\n========================================================')

start = time.time()

# SET PARAMETERS AND LIMITS OF INTEGRATION
kxi = - math.pi / a
kxf = math.pi / a

kyi = - math.pi / a
kyf = math.pi / a

setqx(0.01)

MC = ZMCIntegral.MCintegral(modDs_real,[[kxi,kxf],[kyi,kyf]])

# Setting the zmcintegral parameters
MC.depth = 3
MC.sigma_multiplication = 100
MC.num_trials = 10

result1 = MC.evaluate()
print('Result for qx = ',qx, ': ', result1[0], ' with error: ', result1[1])
print('================================================================')
end = time.time()
print('Computed in ', end-start, ' seconds.')
print('================================================================')
print('================================================================')

start = time.time()

# SET PARAMETERS AND LIMITS OF INTEGRATION
kxi = - math.pi / a
kxf = math.pi / a

kyi = - math.pi / a
kyf = math.pi / a

setqx(0.05)

MC = ZMCIntegral.MCintegral(modDs_real,[[kxi,kxf],[kyi,kyf]])

# Setting the zmcintegral parameters
MC.depth = 3
MC.sigma_multiplication = 100
MC.num_trials = 10

result2 = MC.evaluate()
print('Result for qx = ',qx, ': ', result2[0], ' with error: ', result2[1])
print('================================================================')
end = time.time()
print('Computed in ', end-start, ' seconds.')
print('================================================================')
print('================================================================')

start = time.time()

# SET PARAMETERS AND LIMITS OF INTEGRATION
kxi = - math.pi / a
kxf = math.pi / a

kyi = - math.pi / a
kyf = math.pi / a

setqx(0.1)

MC = ZMCIntegral.MCintegral(modDs_real,[[kxi,kxf],[kyi,kyf]])

# Setting the zmcintegral parameters
MC.depth = 3
MC.sigma_multiplication = 100
MC.num_trials = 10

result3 = MC.evaluate()
print('Result for qx = ',qx, ': ', result3[0], ' with error: ', result3[1])
print('================================================================')
end = time.time()
print('Computed in ', end-start, ' seconds.')
print('================================================================')
print('================================================================')




print('================================================================')
print('kxi  | kxf  | kyi  | kyf  | qx  | qy   | integrand | err')
print('================================================================')

print('%.3f|%.3f|%.3f|%.3f| 0.01 | 0    |%11.8E| %.3E ' % (kxi,kyf, kyi, kyf,result1[0], result1[1]))
print('%.3f|%.3f|%.3f|%.3f| 0.05 | 0    |%11.8E| %.3E ' % (kxi,kyf, kyi, kyf,result2[0], result2[1]))
print('%.3f|%.3f|%.3f|%.3f| 0.1  | 0    |%11.8E| %.3E ' % (kxi,kyf, kyi, kyf,result3[0], result3[1]))

print('================================================================')

toc = time.time()
print('================================================================\n')
print('Process completed successfully!')
print('Total time is ', toc-tic, 'seconds.')
