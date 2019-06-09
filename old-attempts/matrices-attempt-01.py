#!/home/jmcates/miniconda3/envs/zmcint/bin/python
# coding: utf-8

# # Proper interpreter:
# /share/apps/python_shared/3.6.5/bin/python

# # Attempt to evaluate the integral of the function given by Dr. Tse using ZMCintegral
# # The information given is here:
# Mahmoud:
#   "The integrand is Ds(kx,ky,qx,qy)/(2*pi)^3, and the limits of integration are kx=[-pi/a,pi/a],ky=[-pi/a,pi/a] , qx=[-pi/a,pi/a] and qy=[-pi/a,pi/a]."
#   "For qx and qy it is more efficient to use qx=[0.001,pi/a] and qy=0, because of the symmetry of the problem. kx and ky should be as we said before kx=[-pi/a,pi/a],ky=[-pi/a,pi/a]."

# Dr. Tse: 
#   "Hi Jalen, what we need is a plot of the integrated result as a function of qx. My postdoc Mahmoud has a plot for that he obtained previously from another integration method that 
#   we can compare your MC results with. "

# The import statements
import math
from numba import cuda
import ZMCIntegral
import time
import numpy as np
import scipy
import scipy.special
from scipy.integrate import quad

# Start timer
start = time.time()

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

# Ensure that cuda is used
@cuda.jit(device=True)
 # Makes the function below no longer callable, so a test evaluation can not be done with this.

# Function given with slight modification. I replaced all calls to kx, ky, qx, and qy with x[0], x[1], x[2], and x[3] respectively.
# 	This modification effectively "vectorizes" the input.
def Ds(x):
    N = 1
    dds = 0
    ds = 0
    ek = A * (math.sqrt((x[0]) ** 2 + (x[1]) ** 2)) ** 2 + A * (eE0 / hOmg) ** 2
    ekq = A * (math.sqrt((x[0] + x[2]) ** 2 + (x[1] + x[3]) ** 2)) ** 2 + A * (eE0 / hOmg) ** 2
    xk = 2 * A * eE0 * math.sqrt((x[0]) ** 2 + (x[1]) ** 2) / hOmg ** 2
    xkq = 2 * A * eE0 * math.sqrt((x[0] + x[2]) ** 2 + (x[1] + x[3]) ** 2) / hOmg ** 2

    sing = np.arange(-(N - 1) / 2, (N - 1) / 2 + 1, 1)
    taninv1kp = 2 * np.arctan2(Gamm, ek - hOmg / 2 + hOmg * sing)
    taninv1kqp = 2 * np.arctan2(Gamm, ekq - hOmg / 2 + hOmg * sing)
    taninv1km = 2 * np.arctan2(Gamm, ek + hOmg / 2 + hOmg * sing)
    taninv1kqm = 2 * np.arctan2(Gamm, ekq + hOmg / 2 + hOmg * sing)

    lg1kp = complex(0, 1) * np.log(Gamm ** 2 + (ek - hOmg / 2 + hOmg * sing) ** 2)
    lg1kqp = complex(0, 1) * np.log(Gamm ** 2 + (ekq - hOmg / 2 + hOmg * sing) ** 2)
    lg1km = complex(0, 1) * np.log(Gamm ** 2 + (ek + hOmg / 2 + hOmg * sing) ** 2)
    lg1kqm = complex(0, 1) * np.log(Gamm ** 2 + (ekq + hOmg / 2 + hOmg * sing) ** 2)

    ferp = np.heaviside(mu - hOmg / 2 - hOmg * sing, 0)
    ferm = np.heaviside(mu + hOmg / 2 - hOmg * sing, 0)

    dbl = np.arange(-(N - 1), (N - 1) + 1, 1)
    taninv2k = 2 * np.arctan2(Gamm, ek - mu + hOmg * dbl)
    taninv2kq = 2 * np.arctan2(Gamm, ekq - mu + hOmg * dbl)

    lg2k = complex(0, 1) * np.log(Gamm ** 2 + (ek - mu + hOmg * dbl) ** 2)
    lg2kq = complex(0, 1) * np.log(Gamm ** 2 + (ekq - mu + hOmg * dbl) ** 2)

    besk = scipy.special.jv(dbl, xk)
    beskq = scipy.special.jv(dbl, xkq)

    fac1 = ek - ekq + hOmg * dbl
    fac2 = fac1 + 2 * complex(0, 1) * Gamm
    fac3 = fac2 - ek + ekq

    for n in range(0, N):
        for alpha in range(0, N):
            for beta in range(0, N):
                for gamma in range(0, N):
                    for s in range(0, N):
                        for l in range(0, N):
                            p1p = fac1[beta - gamma + N - 1] * (
                                    taninv1kp[alpha] - taninv2k[s + alpha] - lg1kp[alpha] + lg2k[s + alpha])
                            p2p = fac2[alpha - gamma + N - 1] * (
                                    taninv1kp[beta] - taninv2k[s + beta] + lg1kp[beta] - lg2k[s + beta])
                            p3p = fac3[alpha - beta + N - 1] * (
                                    -taninv1kqp[gamma] + taninv2kq[s + gamma] - lg1kqp[gamma] + lg2kq[
                                s + gamma])

                            p1m = fac1[beta - gamma + N - 1] * (
                                    taninv1km[alpha] - taninv2k[s + alpha] - lg1km[alpha] + lg2k[s + alpha])

                            p2m = fac2[alpha - gamma + N - 1] * (
                                    taninv1km[beta] - taninv2k[s + beta] + lg1km[beta] - lg2k[s + beta])

                            p3m = fac3[alpha - beta + N - 1] * (
                                    -taninv1kqm[gamma] + taninv2kq[s + gamma] - lg1kqm[gamma] + lg2kq[
                                s + gamma])

                            d1 = -2 * complex(0, 1) * fac1[beta - gamma + N - 1] * fac2[alpha - gamma + N - 1] * \
                                 fac3[
                                     alpha - beta + N - 1]

                            omint1p = ferp[s] * ((p1p + p2p + p3p) / d1)

                            omint1m = ferm[s] * ((p1m + p2m + p3m) / d1)

                            bess1 = beskq[gamma - n + N - 1] * beskq[gamma - l + N - 1] * besk[beta - l + N - 1] * besk[
                                beta - s + N - 1] * besk[alpha - s + N - 1] * besk[alpha - n + N - 1]

                            grgl = bess1 * (omint1p - omint1m)

                            pp1p = fac1[alpha - beta + N - 1] * (
                                    -taninv1kqp[gamma] + taninv2kq[s + gamma] - lg1kqp[gamma] + lg2kq[
                                s + gamma])

                            pp2p = fac2[alpha - gamma + N - 1] * (
                                    -taninv1kqp[beta] + taninv2kq[s + beta] + lg1kqp[beta] - lg2kq[
                                s + beta])

                            pp3p = fac3[beta - gamma + N - 1] * (
                                    taninv1kp[alpha] - taninv2k[s + alpha] - lg1kp[alpha] + lg2k[s + alpha])

                            pp1m = fac1[alpha - beta + N - 1] * (
                                    -taninv1kqm[gamma] + taninv2kq[s + gamma] - lg1kqm[gamma] + lg2kq[
                                s + gamma])

                            pp2m = fac2[alpha - gamma + N - 1] * (
                                    -taninv1kqm[beta] + taninv2kq[s + beta] + lg1kqm[beta] - lg2kq[
                                s + beta])

                            pp3m = fac3[beta - gamma + N - 1] * (
                                    taninv1km[alpha] - taninv2k[s + alpha] - lg1km[alpha] + lg2k[s + alpha])

                            d2 = -2 * complex(0, 1) * fac1[alpha - beta + N - 1] * fac2[alpha - gamma + N - 1] * \
                                 fac3[
                                     beta - gamma + N - 1]

                            omint2p = ferp[s] * ((pp1p + pp2p + pp3p) / d2)

                            omint2m = ferm[s] * ((pp1m + pp2m + pp3m) / d2)

                            bess2 = beskq[gamma - n + N - 1] * beskq[gamma - s + N - 1] * beskq[beta - s + N - 1] * \
                                    beskq[beta - l + N - 1] * besk[alpha - l + N - 1] * besk[alpha - n + N - 1]

                            glga = bess2 * (omint2p - omint2m)

                            dds = dds + 2 * Gamm * (grgl + glga)
    return dds

# # As mentioned above, the function is no longer callable. This test is invalid.
# testeval = Ds(0.1, 0.1, 0.01, 0)
# Test evaluation
# print('\n========================================================')
# print('The function is evaluated at (0.1,0.1,0.1,0) to be:')
# print(testeval)
# print('\n========================================================')

# Mahmoud:
#   "The integrand is Ds(kx,ky,qx,qy)/(2*pi)^3, and the limits of integration are kx=[-pi/a,pi/a],ky=[-pi/a,pi/a] , qx=[-pi/a,pi/a] and qy=[-pi/a,pi/a]."
#   "For qx and qy it is more efficient to use qx=[0.001,pi/a] and qy=0, because of the symmetry of the problem. kx and ky should be as we said before kx=[-pi/a,pi/a],ky=[-pi/a,pi/a]."


# Introducing variables for limits of integration
kxi = - math.pi / a
kxf = math.pi / a

kyi = - math.pi / a
kyf = math.pi / a

qxi = 0.001
qxf = math.pi / a

qyi = 0
qyf = 0

# Print statements to show the limits of integration
print('The limits of integration:')
print('  kx = (', kxi, ', ', kxf, ')')
print('  ky = (', kyi, ', ', kyf, ')')
print('  qx = (', qxi, ', ', qxf, ')')
print('  qy = (', qyi, ', ', qyf, ')')

MC = ZMCIntegral.MCintegral(Ds,[[kxi,kxf],[kyi,kyf],[qxi,qxf],[qyi,qyf]])

# Setting the zmcintegral parameters
MC.depth = 2
MC.sigma_multiplication = 5
MC.num_trials = 5
MC.available_GPU=[0]



print('\n========================================================')
print('depth = ', MC.depth)
print('sigma_multiplication = ', MC.sigma_multiplication)
print('num_trials = ', MC.num_trials)
print('available_GPU = ', MC.available_GPU)

# Evaluating the integral
result = MC.evaluate()

end = time.time()

print('\n========================================================')
print('Integration is complete!')
print('\n========================================================')
print('Result: ', result[0])
print('std.  : ', result[1])
print('Computed in ', end-start, ' seconds.')
print('\n========================================================')
