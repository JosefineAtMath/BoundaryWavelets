# -*- coding: utf-8 -*-
"""
This module is used for calculations of the orthonormalization matrix for
the boundary wavelets.

The BoundaryWavelets.py package is licensed under the MIT "Expat" license.

Copyright (c) 2019: Josefine Holm and Steffen L. Nielsen.
"""
# =============================================================================
# Imports
# =============================================================================
import numpy as np
from scipy.integrate import simps
import boundwave.BoundaryWavelets as BW


# =============================================================================
# Functions
# =============================================================================
def Integral(J, k, l, WaveletCoef, phi):
    r'''
    This function calculates the integral (16) numerically.

    INPUT:
        J : int
            The scale.
        k : int
            The translation for the first function.
        l : int
            The translation for the second function.
        WaveletCoef : numpy.float64
            The wavelet coefficients, must sum to :math:`\sqrt{2}`.
            For Daubechies 2 they can be found using
            `np.flipud(pywt.Wavelet('db2').dec_lo)`.
        phi : numpy.float64
            The phi function, can be made with
            `pywt.Wavelet(wavelet).wavefun(level=15)`.
    OUTPUT:
        out : int
            The value of the integral.

    '''

    a = int(len(WaveletCoef)/2)
    OneStep = len(phi)//(2*a-1)
    phiNorm = np.linalg.norm(BW.DownSample(phi, 0, OneStep, J))
    phi1 = BW.DownSample(phi, k, OneStep, J)/phiNorm
    phi2 = BW.DownSample(phi, l, OneStep, J)/phiNorm
    phiProd = phi1*phi2
    Integ = simps(phiProd)
    return Integ


def M_AlphaBeta(alpha, beta, J, WaveletCoef, InteMatrix, Side):
    r'''
    This function calculates an entry in the martix M (15).

    INPUT:
        alpha : int
            alpha
        beta : int
            beta
        J : int
            The scale.
        WaveletCoef : numpy.float64
            The wavelet coefficients, must sum to :math:`\sqrt{2}`. For
            Daubechies 2 they can be found using
            `np.flipud(pywt.Wavelet('db2').dec_lo`).
        InteMatrix : numpy.float64
            A matrix with the values for the integrals calculated with
            the function integral() for k and l in the interval
            [-2*a+2,0] or [2**J-2*a+1,2**J-1].
        Side : str
            `'L'` for left interval boundary and `'R'` for right
            interval boundary.
    OUTPUT:
        M : numpy.float64
            Entry (alpha,beta) of the martix M

    '''

    a = int(len(WaveletCoef)/2)
    Moment = BW.Moments(WaveletCoef, a-1)
    M = 0
    if Side == 'L':
        Interval = range(-2*a+2, 1)
        i = 0
        for k in Interval:
            j = 0
            for m in Interval:
                M += (BW.InnerProductPhiX(alpha, 0, k, Moment) *
                      BW.InnerProductPhiX(beta, 0, m, Moment) *
                      InteMatrix[i, j])
                j += 1
            i += 1
    elif Side == 'R':
        Interval = range(2**J-2*a+1, 2**J)
        i = 0
        for k in Interval:
            j = 0
            for m in Interval:
                M += (BW.InnerProductPhiX(alpha, 0, k, Moment) *
                      BW.InnerProductPhiX(beta, 0, m, Moment) *
                      InteMatrix[i, j] * 2**(-J*(alpha+beta)))
                j += 1
            i += 1
    else:
        print('You must choose a side')

    return M


def OrthoMatrix(J, WaveletCoef, phi):
    r'''
    This function findes the orthogonality matrix A. First use the functions
    M_AlphaBeta() and integral() to make the matrix M. Then does a cholesky
    decomposition, which is then inverted.

    INPUT:
        J : int
            The scale.
        WaveletCoef : numpy.float64
            The wavelet coefficients, must sum to
            :math:`\sqrt{2}`. For Daubechies 2 they can be found using
            `np.flipud(pywt.Wavelet('db2').dec_lo)`.
        phi : numpy.float64
            The phi function, can be made with
            `pywt.Wavelet(wavelet).wavefun(level=15)`.
    OUTPUT:
        AL : numpy.float64
            Left orthonormalisation matrix; to be used in
            :py:func:`boundwave.BoundaryWavelets.BoundaryWavelets` or
            :py:func:`boundwave.FourierBoundaryWavelets.FourierBoundaryWavelets`.
        AR : numpy.float64
            Right orthonormalisation matrix; to be used in
            :py:func:`boundwave.BoundaryWavelets.BoundaryWavelets` or
            :py:func:`boundwave.FourierBoundaryWavelets.FourierBoundaryWavelets`.

    '''

    a = int(len(WaveletCoef)/2)
    ML = np.zeros((a, a))
    MR = np.zeros((a, a))
    InteL = np.zeros((2*a-1, 2*a-1))
    k = 0
    for i in range(-2*a+2, 1):
        m = 0
        for j in range(-2*a+2, i+1):
            InteL[k, m] = Integral(J, i, j, WaveletCoef, phi)
            InteL[m, k] = InteL[k, m]
            m += 1
        k += 1
    InteR = np.zeros((2*a-1, 2*a-1))
    k = 0
    for i in range(2**J-2*a+1, 2**J):
        m = 0
        for j in range(2**J-2*a+1, i+1):
            InteR[k, m] = Integral(J, i, j, WaveletCoef, phi)
            InteR[m, k] = InteR[k, m]
            m += 1
        k += 1
    for i in range(a):
        for j in range(i+1):
            ML[i, j] = M_AlphaBeta(i, j, J, WaveletCoef, InteL, 'L')
            ML[j, i] = ML[i, j]
    for i in range(a):
        for j in range(i+1):
            MR[i, j] = M_AlphaBeta(i, j, J, WaveletCoef, InteR, 'R')
            MR[j, i] = MR[i, j]
    h = 2**(J * np.arange(a))
    CL = np.linalg.cholesky(ML)
    AL = 2**(J/2) * np.dot(np.linalg.inv(CL), np.diag(h))
    CR = np.linalg.cholesky(MR)
    U, S, V = np.linalg.svd(CR)
    AR = 2**(J/2)*np.dot(np.dot(np.transpose(V), np.diag(1/S)),
                         np.transpose(U))
    return AL, AR
