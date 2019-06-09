
# coding: utf-8

# # Test to ensure modifications to function do not change evaluated value
# 
# Beginning modification of the function to handle case where N is not equal to 1.


import math
from numba import cuda
import ZMCIntegral
import time
import numpy as np
import helpers
from numba.extending import overload
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


@cuda.jit(device=True)
def my_Besselv(v, z):
    # WILL NOT WORK IF v IS NOT AN INTEGER
    # Conditional to handle case of negative v.
    if(v < 0):
        v = abs(v)
        resultsign = (-1) ** v
    else:
        resultsign = 1
    result = 0    
    # Loop to construct Bessel series sum.
    for n in range(0,20):
        sign = (-1)**n
        exp = 2 * n + v
        term = z ** exp
        r = n + v + 1
        denom = math.gamma(r)
        denom = denom * math.gamma(n+1)
        denom = denom * (2 ** exp)
        term = term / denom * sign
        # print('for ', n, ': ',term)
        result = result + term
        
    return result * resultsign
    
@cuda.jit(device=True)    
def Dslist1(ek, ekq, N):
    taninv1kp = (complex(1,1))
    taninv1kqp = (complex(1,1))
    taninv1km = (complex(1,1))
    taninv1kqm = (complex(1,1))

    lg1kp = (complex(1,1))
    lg1kqp = (complex(1,1))
    lg1km = (complex(1,1))
    lg1kqm = (complex(1,1))
            
    ferp = (complex(1,1))
    ferm = (complex(1,1))
    i = -(N - 1) / 2
    while(i < ((N - 1) / 2 + 1)):
        taninv1kp.append(2 * math.atan2(Gamm, ek - hOmg / 2 + hOmg * i))
        taninv1kqp.append(2 * math.atan2(Gamm, ekq - hOmg / 2 + hOmg * i))
        taninv1km.append(2 * math.atan2(Gamm, ek + hOmg / 2 + hOmg * i))
        taninv1kqm.append(2 * math.atan2(Gamm, ekq + hOmg / 2 + hOmg * i))

        lg1kp.append(complex(0, 1) * math.log(Gamm ** 2 + (ek - hOmg / 2 + hOmg * i) ** 2))
        lg1kqp.append(complex(0, 1) * math.log(Gamm ** 2 + (ekq - hOmg / 2 + hOmg * i) ** 2))
        lg1km.append(complex(0, 1) * math.log(Gamm ** 2 + (ek + hOmg / 2 + hOmg * i) ** 2))
        lg1kqm.append(complex(0, 1) * math.log(Gamm ** 2 + (ekq + hOmg / 2 + hOmg * i) ** 2))
               
        ferp.append(helpers.my_heaviside(mu - hOmg / 2 - hOmg * i))
        ferm.append(helpers.my_heaviside(mu + hOmg / 2 - hOmg * i))
        i = i + 1
    
    taninv1kp.remove(taninv1kp[0])
    taninv1kqp.remove(taninv1kqp[0])
    taninv1km.remove(taninv1km[0])
    taninv1kqm.remove(taninv1kqm[0])

    lg1kp.remove(lg1kp[0])
    lg1kqp.remove(lg1kqp[0])
    lg1km.remove(lg1km[0])
    lg1kqm.remove(lg1kqm[0])
            
    ferp.remove(ferp[0])
    ferm.remove(ferm[0])

    firstList = (taninv1kp, taninv1kqp, taninv1km, taninv1kqm, lg1kp, lg1kqp, lg1km, lg1kqm, ferp, ferm)
    return firstList



@cuda.jit(device=True)        
def Dslist2(ek, ekq, xk, xkq, N):
    # size_dbl = 2 * N - 1

    taninv2k = (complex(1,1))
    taninv2kq = (complex(1,1))

    lg2k = (complex(1,1))
    lg2kq = (complex(1,1))

    besk = (complex(1,1))
    beskq = (complex(1,1))

    fac1 = (complex(1,1))
    fac2 = (complex(1,1))
    fac3 = (complex(1,1))

    for i in range(-(N - 1), N, 1):
        xi = hOmg * i
        zeta = ek - mu + xi
        eta = ekq - mu + xi

        taninv2ki = 2 * math.atan2(Gamm, zeta)
        taninv2kqi = 2 * math.atan2(Gamm, eta)

        taninv2k.append(taninv2ki)
        taninv2kq.append(taninv2kqi)

        lg2ki = complex(0, 1) * math.log(Gamm ** 2 + (zeta) ** 2)
        lg2kqi = complex(0, 1) * math.log(Gamm ** 2 + (eta) ** 2)

        lg2k.append(lg2ki)
        lg2kq.append(lg2kqi)
        
        beski = my_Besselv(i, xk)
        beskqi = my_Besselv(i, xkq)

        besk.append(beski)
        beskq.append(beskqi)

        fac1i = ek - ekq + xi
        fac2i = fac1i + 2 * complex(0, 1) * Gamm
        fac3i = fac2i - ek + ekq

        fac1.append(fac1i)
        fac2.append(fac2i)
        fac3.append(fac3i)

    taninv2k.remove(taninv2k[0])
    taninv2kq.remove(taninv2kq[0])

    lg2k.remove(lg2k[0])
    lg2kq.remove(lg2kq[0])

    besk.remove(besk[0])
    beskq.remove(beskq[0])

    fac1.remove(fac1[0])
    fac2.remove(fac2[0])
    fac3.remove(fac3[0])

    secondList = (taninv2k, taninv2kq, lg2k, lg2kq, besk, beskq, fac1, fac2, fac3)
    return secondList



# @cuda.jit('float64(float64[:])',device=True)        
def modDsN2(x):
    N = 3
    dds = 0
    # ds = 0 # UNUSED
    qx = helpers.getqx()
    ek = A * (math.sqrt((x[0]) ** 2 + (x[1]) ** 2)) ** 2 + A * (eE0 / hOmg) ** 2
    ekq = A * (math.sqrt((x[0] + qx) ** 2 + (x[1] + 0) ** 2)) ** 2 + A * (eE0 / hOmg) ** 2
    xk = 2 * A * eE0 * math.sqrt((x[0]) ** 2 + (x[1]) ** 2) / hOmg ** 2
    xkq = 2 * A * eE0 * math.sqrt((x[0] + qx) ** 2 + (x[1] + 0) ** 2) / hOmg ** 2
    
    firstList = Dslist1(ek, ekq, N)
    secondList = Dslist2(ek, ekq, xk, xkq, N)

    for n in range(0, N):
        for alpha in range(0, N):
            for beta in range(0, N):
                for gamma in range(0, N):
                    for s in range(0, N):
                        for l in range(0, N):
                            p1p = secondList[6][beta - gamma + N - 1] * (
                                    firstList[0][alpha] - secondList[0][s + alpha] - firstList[4][alpha] + secondList[2][s + alpha])
                            p2p = secondList[7][alpha - gamma + N - 1] * (
                                    firstList[0][beta] - secondList[0][s + beta] + firstList[4][beta] - secondList[2][s + beta])
                            p3p = secondList[8][alpha - beta + N - 1] * (
                                    -firstList[1][gamma] + secondList[1][s + gamma] - firstList[5][gamma] + secondList[3][
                                s + gamma])

                            p1m = secondList[6][beta - gamma + N - 1] * (
                                    firstList[2][alpha] - secondList[0][s + alpha] - firstList[6][alpha] + secondList[2][s + alpha])

                            p2m = secondList[7][alpha - gamma + N - 1] * (
                                    firstList[2][beta] - secondList[0][s + beta] + firstList[6][beta] - secondList[2][s + beta])

                            p3m = secondList[8][alpha - beta + N - 1] * (
                                    -firstList[3][gamma] + secondList[1][s + gamma] - firstList[7][gamma] + secondList[3][
                                s + gamma])

                            d1 = -2 * complex(0, 1) * secondList[6][beta - gamma + N - 1] * secondList[7][alpha - gamma + N - 1] *                                  secondList[8][
                                     alpha - beta + N - 1]

                            omint1p = firstList[8][s] * ((p1p + p2p + p3p) / d1)

                            omint1m = firstList[9][s] * ((p1m + p2m + p3m) / d1)

                            bess1 = secondList[5][gamma - n + N - 1] * secondList[5][gamma - l + N - 1] * secondList[4][beta - l + N - 1] * secondList[4][
                                beta - s + N - 1] * secondList[4][alpha - s + N - 1] * secondList[4][alpha - n + N - 1]

                            grgl = bess1 * (omint1p - omint1m)

                            pp1p = secondList[6][alpha - beta + N - 1] * (
                                    -firstList[1][gamma] + secondList[1][s + gamma] - firstList[5][gamma] + secondList[3][
                                s + gamma])

                            pp2p = secondList[7][alpha - gamma + N - 1] * (
                                    -firstList[1][beta] + secondList[1][s + beta] + firstList[5][beta] - secondList[3][
                                s + beta])

                            pp3p = secondList[8][beta - gamma + N - 1] * (
                                    firstList[0][alpha] - secondList[0][s + alpha] - firstList[4][alpha] + secondList[2][s + alpha])

                            pp1m = secondList[6][alpha - beta + N - 1] * (
                                    -firstList[3][gamma] + secondList[1][s + gamma] - firstList[7][gamma] + secondList[3][
                                s + gamma])

                            pp2m = secondList[7][alpha - gamma + N - 1] * (
                                    -firstList[3][beta] + secondList[1][s + beta] + firstList[7][beta] - secondList[3][
                                s + beta])

                            pp3m = secondList[8][beta - gamma + N - 1] * (
                                    firstList[2][alpha] - secondList[0][s + alpha] - firstList[6][alpha] + secondList[2][s + alpha])

                            d2 = -2 * complex(0, 1) * secondList[6][alpha - beta + N - 1] * secondList[7][alpha - gamma + N - 1] *                                  secondList[8][
                                     beta - gamma + N - 1]

                            omint2p = firstList[8][s] * ((pp1p + pp2p + pp3p) / d2)

                            omint2m = firstList[9][s] * ((pp1m + pp2m + pp3m) / d2)

                            bess2 = secondList[5][gamma - n + N - 1] * secondList[5][gamma - s + N - 1] * secondList[5][beta - s + N - 1] *                                     secondList[5][beta - l + N - 1] * secondList[4][alpha - l + N - 1] * secondList[4][alpha - n + N - 1]

                            glga = bess2 * (omint2p - omint2m)

                            dds = dds + 2 * Gamm * (grgl + glga)
                            
                            #DEBUG
                            # print('moddds = ', dds)
                            # print('modbess1=',  bess1)
    return dds.real


kxi = - math.pi / a
kxf = math.pi / a

kyi = - math.pi / a
kyf = math.pi / a

helpers.setqx(0.01)

MC = ZMCIntegral.MCintegral(modDsN2,[[kxi,kxf],[kyi,kyf]])

# Setting the zmcintegral parameters
MC.depth = 3
MC.sigma_multiplication = 100
MC.num_trials = 10

start = time.time()

result = MC.evaluate()
print('Result for qx = 0.01 :', result[0], ' with error: ', result[1])
print('================================================================')

end = time.time()
print('Computed in ', end-start, ' seconds.')
print('================================================================')
