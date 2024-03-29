import numpy as np
import math
import scipy
import scipy.special
import time
from scipy.integrate import quad
import multiprocessing
from numba import cuda
import numba as nb

start = time.time()
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


# print(shift)

def Ds(kx, ky, qx, qy):
    N = 1
    dds = 0
    ds = 0
    ek = A * (math.sqrt((kx) ** 2 + (ky) ** 2)) ** 2 + A * (eE0 / hOmg) ** 2
    ekq = A * (math.sqrt((kx + qx) ** 2 + (ky + qy) ** 2)) ** 2 + A * (eE0 / hOmg) ** 2
    xk = 2 * A * eE0 * math.sqrt((kx) ** 2 + (ky) ** 2) / hOmg ** 2
    xkq = 2 * A * eE0 * math.sqrt((kx + qx) ** 2 + (ky + qy) ** 2) / hOmg ** 2

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


print(Ds(0.1, 0.1, 0.01, 0))

end = time.time()
print(end - start)
N = 3
sing = np.linspace(-1, 1, N)

ts = np.sin(sing)


def myfunc(a, b, c):
    x = np.sin(a - b + c)
    return x

# vfunc = np.vectorize(myfunc)
# x = [[1, 2, 3], [1, 2, 3]]
# r = vfunc(x, 1, x)
