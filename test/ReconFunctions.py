# -*- coding: utf-8 -*-
"""
This is a module which contains reconstruction algorithms for the Daubechies
wavelets.

The BoundaryWavelets.py package is licensed under the MIT "Expat"
Copyright (c) 2019: Josefine Holm and Steffen L. Nielsen.
"""
# =============================================================================
# Imports
# =============================================================================
import numpy as np
import pywt
import BoundaryWavelets as BW
import Orthonormal as Ot


# =============================================================================
# Functions
# =============================================================================
def ReconBoundary(WaveletCoef, J, Wavelet, phi, N=15):
    '''
    This function reconstructs a 1D signal in time from its wavelet
    coefficients, using boundary wavelets at the edge.

    INPUT:
        WaveletCoef:
            numpy.float64: The wavelet coefficients, must sum to sqrt(2).
            For Daubechies 2 they can be found using
            `np.flipud(pywt.Wavelet('db2').dec_lo)`.
        J:
            int: The scale of the wavelet
        wavelet:
            str: The name of the wavelet to use. For instance 'db2'.
        N=15:
            int: The number of iterations to use in the cascade algorithm when
            estimating phi.
    OUTPUT:
        x:
            numpy.float64: The reconstructed signal, the length of the signal
            is 2**(N-j)*len(W).
    '''
    h = np.flipud(pywt.Wavelet(Wavelet).dec_lo)
    AL, AR = Ot.OrthoMatrix(J, h, phi)
    Q = BW.BoundaryWavelets(phi, J, h, AL=AL, AR=AR)
    phi = BW.DownSample(phi, 0, 2**N, J, zero=False)
    OneStep = 2**(N-J)
    m = np.shape(Q)[0]
    a = np.shape(Q)[1]//2
    Boundary = np.transpose(Q)
    x = np.zeros(OneStep*len(WaveletCoef), dtype=complex)
    for i in range(a):
        x[:m] += WaveletCoef[i]*Boundary[i]
    k = 1
    for i in range(a, len(WaveletCoef)-a):
        x[k*OneStep:k*OneStep+len(phi)] += WaveletCoef[i]*phi*2**(-(J)/2)
        k += 1
    for i in range(a):
        x[-m:] += WaveletCoef[-i-1]*Boundary[-i-1]
    return x


def ReconMirror(WaveletCoef, J, Wavelet, phi, N=15):
    '''
    This function reconstructs a 1D signal in time from its wavelet
    coefficients, using mirroring of the signal at the edge.

    INPUT:
        WaveletCoef:
            numpy.float64: The wavelet coefficients, must sum to sqrt(2).
            For Daubechies 2 they can be found using
            `np.flipud(pywt.Wavelet('db2').dec_lo)`.
        J:
            int: The scale of the wavelet
        wavelet:
            str: The name of the wavelet to use. For instance 'db2'.
        N=15:
            int: The number of iterations to use in the cascade algorithm when
            estimating phi.
    OUTPUT:
        x:
            numpy.float64: The reconstructed signal, the length of the signal
            is 2**(N-j)*len(W).
    '''

    phi = BW.DownSample(phi, 0, 2**N, J, zero=False)
    OneStep = 2**(N-J)
    a = int(len(phi)/OneStep)-1
    x = np.zeros(OneStep*len(WaveletCoef)+(a)*OneStep, dtype=complex)
    for i in range(len(WaveletCoef)):
        x[i*OneStep:i*OneStep+len(phi)] += WaveletCoef[i]*phi*2**(-(J)/2)
    x = x[OneStep*(a):-OneStep*(a)]
    return x


def DecomBoundary(Signal, J, Wavelet, phi, N=15):
    '''
    This function makes a wavelet decomposition of a 1D signal in time, using
    boundary wavelets at the edge.

    INPUT:
        WaveletCoef:
            numpy.float64: The wavelet coefficients, must sum to sqrt(2).
            For Daubechies 2 they can be found using
            `np.flipud(pywt.Wavelet('db2').dec_lo)`.
        J:
            int: The scale of the wavelet
        wavelet:
            str: The name of the wavelet to use. For instance 'db2'.
        N=15:
            int: The number of iterations to use in the cascade algorithm when
            estimating phi.
    OUTPUT:
        x:
            numpy.float64: The decomposition.
    '''
    WaveletCoef = np.flipud(pywt.Wavelet(Wavelet).dec_lo)
    a = int(len(WaveletCoef)/2)
    AL, AR = Ot.OrthoMatrix(J, WaveletCoef, phi)
    Boundary = BW.BoundaryWavelets(phi, J, WaveletCoef, AL=AL, AR=AR)
    x = np.zeros(2**J)
    for i in range(a):
        x[i] = np.inner(Boundary[:, i], Signal)
    for i in range(1, 2**J-2*a+1):
        x[i-1+a] = np.inner(BW.DownSample(phi, i, 2**N, J, zero=True) *
                            np.sqrt(2**J), Signal)
    for i in range(a):
        x[-1-i] = np.inner(Boundary[:, -1-i], Signal)
    x /= len(Signal)
    return x


def DecomMirror(Signal, J, Wavelet, phi, N):
    '''
    This function makes a wavelet decomposition of a 1D signal in time, using
    mirroring af the signal at the edge. \n
    INPUT:
        WaveletCoef:
            numpy.float64: The wavelet coefficients, must sum to sqrt(2).
            For Daubechies 2 they can be found using
            `np.flipud(pywt.Wavelet('db2').dec_lo)`.
        J:
            int: The scale of the wavelet
        wavelet:
            str: The name of the wavelet to use. For instance 'db2'.
        N=15:
            int: The number of iterations to use in the cascade algorithm when
            estimating phi.
    OUTPUT:
        x:
            numpy.float64: The decomposition.
    '''
    OneStep = 2**(N-J)
    WaveletCoef = np.flipud(pywt.Wavelet(Wavelet).dec_lo)
    a = int(len(WaveletCoef)/2)
    x = np.zeros(2**J+(2*a-2))
    for i in range(2*a-2):
        phi1 = BW.DownSample(phi, 0, 2**N, J, zero=False)*np.sqrt(2**J)
        Signal1 = np.concatenate((np.flipud(Signal[:OneStep*(i+1)]), Signal))
        Signal1 = Signal1[:len(phi1)]
        x[i] = np.inner(phi1, Signal1)
    for i in range(2**J-2*a+2):
        x[i+2*a-2] = np.inner(BW.DownSample(phi, i, 2**N, J, zero=True) *
                              np.sqrt(2**J), Signal)
    for i in range(2*a-2):
        phi1 = BW.DownSample(phi, 0, 2**N, J, zero=False)*np.sqrt(2**J)
        Signal1 = np.concatenate((Signal, np.flipud(Signal[-OneStep*(i+1):])))
        Signal1 = Signal1[-len(phi1):]
        x[2**J+i] = np.inner(phi1, Signal1)
    x /= len(Signal)
    return x
