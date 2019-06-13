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

# Define constants in function

mu = 0.1  # Fermi-level
hOmg = 0.5  # Photon energy eV
a = 4  # AA
A = 4  # hbar^2/(2m)=4 evAA^2 (for free electron mass)
rati = 0.3  # ratio
eE0 = rati * ((hOmg) ** 2) / (2 * np.sqrt(A * mu))
# print(eE0)
Gamm = 0.005  # Gamma in eV.
KT = 1 * 10 ** (-6)
shift = A * (eE0 / hOmg) ** 2


# Function given with slight modification. I replaced all calls to kx, ky, qx, and qy with x[0], x[1], x[2], and x[3] respectively. This modification effectively "vectorizes" the input.

sing = np.array([0.])

@cuda.jit(device=True)
def my_heaviside(z): 
    # Wrote this Heaviside expression with it cast in cuda to avoid error below.
    if z <= 0 :
	    return 0
    else :
	    return 1

@cuda.jit(device=True)
def my_Bessel(z):
    # CHANGE FOR v4: Changing polynomial evaluation for efficiency (SEE Horner's Algorithm). Extending number of terms.
    # Carrying the series to eight terms ensures that the error in the series is < machine_epsilon when z < 1. 
    # Approximately: z1 <~ 2.1, z2 <~ 3.33  implies  error <~ 2.15E-6
    val = z**2 / 4 * (-1 + z**2 / 16 * (1 + z**2 / 36 * (-1 + z**2 / 64 * (1 + z**2 / 100 * (-1 + z**2 / 144 * (1 + z**2 / 196 * (-1 + z**2 / 256)))))))
    return val + 1



@cuda.jit(device=True)
def modDs_real(x):
    dds = 0
    ek = A * (math.sqrt((x[0]) ** 2 + (x[1]) ** 2)) ** 2 + A * (eE0 / hOmg) ** 2
    ekq = A * (math.sqrt((x[0] + x[2]) ** 2 + (x[1] + x[3]) ** 2)) ** 2 + A * (eE0 / hOmg) ** 2
    xk = 2 * A * eE0 * math.sqrt((x[0]) ** 2 + (x[1]) ** 2) / hOmg ** 2
    xkq = 2 * A * eE0 * math.sqrt((x[0] + x[2]) ** 2 + (x[1] + x[3]) ** 2) / hOmg ** 2

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

    ferp = my_heaviside(heavi1)
    ferm = my_heaviside(heavi2)

    taninv2k = 2 * math.atan2(Gamm, ek - mu)
    taninv2kq = 2 * math.atan2(Gamm, ekq - mu)

    lg2k = complex(0, 1) * math.log(Gamm ** 2 + (ek - mu) ** 2)
    lg2kq = complex(0, 1) * math.log(Gamm ** 2 + (ekq - mu) ** 2)

    besk = my_Bessel(xk)
    beskq = my_Bessel(xkq)

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

    dds = dds + 2 * Gamm * (grgl + glga)

    return dds.real / (8*math.pi**3)


# # Mahmoud:
#  "The integrand is Ds(kx,ky,qx,qy)/(2*pi)^3, and the limits of integration are kx=[-pi/a,pi/a],ky=[-pi/a,pi/a] , qx=[-pi/a,pi/a] and qy=[-pi/a,pi/a]."
#
# "For qx and qy it is more efficient to use qx=[0.001,pi/a] and qy=0, because of the symmetry of the problem. kx and ky should be as we said before kx=[-pi/a,pi/a],ky=[-pi/a,pi/a]."


# Introducing suggested values of integration.

kxi = - math.pi / a
kxf = math.pi / a

kyi = - math.pi / a
kyf = math.pi / a

qxi = 0.001
qxf = math.pi / a

qyi = 0
qyf = 0


print('\n========================================================')
print('\nThe limits of integration:')
print('  kx = (', kxi, ', ', kxf, ')')
print('  ky = (', kyi, ', ', kyf, ')')
print('  qx = (', qxi, ', ', qxf, ')')
print('  qy = (', qyi, ', ', qyf, ')')

# Creating the ZMCintegral object for evaluation.

MC = ZMCIntegral.MCintegral(modDs_real,[[kxi,kxf],[kyi,kyf],[qxi,qxf],[qyi,qyf]])

# Setting the zmcintegral parameters
MC.depth = 6
MC.sigma_multiplication = 10**10
MC.num_trials = 15
MC.available_GPU=[0]


print('\n========================================================')
print('\ndepth = ', MC.depth)
print('sigma_multiplication = ', MC.sigma_multiplication)
print('num_trials = ', MC.num_trials)
print('available_GPU = ', MC.available_GPU)
print('\n========================================================')


# # Evaluating integral:

start = time.time()
result = MC.evaluate()
end = time.time()

print('\n========================================================')
print('\nThe limits of integration:')
print('  kx = (', kxi, ', ', kxf, ')')
print('  ky = (', kyi, ', ', kyf, ')')
print('  qx = (', qxi, ', ', qxf, ')')
print('  qy = (', qyi, ', ', qyf, ')')
print('\n========================================================')
print('\ndepth = ', MC.depth)
print('sigma_multiplication = ', MC.sigma_multiplication)
print('num_trials = ', MC.num_trials)
print('available_GPU = ', MC.available_GPU)
print('\n========================================================')
print('\n========================================================')
print('Integration is complete!')
print('\n========================================================')
print('Result: ', result[0])
print('std.  : ', result[1])
print('Computed in ', end-start, ' seconds.')
print('\n========================================================')


