# -*- coding: utf-8 -*-
"""
This module is used for calculations of the boundary wavelets in the time
domain.

The BoundaryWavelets.py package is licensed under the MIT "Expat" license.

Copyright (c) 2019: Josefine Holm and Steffen L. Nielsen.
"""
# =============================================================================
# Imports
# =============================================================================
import numpy as np
from scipy.special import binom


# =============================================================================
# Functions
# =============================================================================
def DownSample(x, Shift, N, J, zero=True):
    """
    This is a function which dilates a signal. Make sure there are
    enough samples for it to make sense.

    INPUT:
        x : numpy.float64
            1d numpy array, signal to be dilated.
        Shift : int
            Time shift before dilation (for now it only supports
            integers).
        N : int
            Number of samples per 'time' unit.
        J : int
            The scale to make. Non-negative integer.
        zero=True : bool
            If true, it concatenates zeros on the signal to retain the
            original length.
    OUTPUT:
        y : numpy.float64
            The scaled version of the scaling function.

    """
    if J == 0:
        if Shift < 0:
            x1 = np.concatenate((x[-Shift*N:], np.zeros(N-len(x[-Shift*N:]))))
        else:
            x1 = np.concatenate((np.zeros(Shift*N), x))
        return x1[:N]
    if Shift <= 0:
        x1 = x[-Shift*N:]
        xhat = np.sqrt(2**J)*x1[::2**J]
        if zero:
            y = np.concatenate((xhat, np.zeros(N-len(xhat))))
        else:
            y = xhat
    else:
        x1 = np.concatenate((np.zeros(Shift*N), x))
        xhat = np.sqrt(2**J)*x1[::2**J]
        if len(xhat) > N:
            y = xhat[:N]
        elif zero:
            y = np.concatenate((xhat, np.zeros(N-len(xhat))))
        else:
            y = xhat
    return y


def Moments(WaveletCoef, n):
    '''
    This function calculates the moments of phi up to power n, i.e. <x**l,phi>,
    for 0<=l<=n.

    INPUT:
        WaveletCoef : numpy.float64
            The wavelet coefficients, must sum to :math:`\sqrt{2}`.
            For Daubechies 2 they can be found using
            `np.flipud(pywt.Wavelet('db2').dec_lo)`.
        n : int
            The highest power moment to calculate.
    OUTPUT:
        moments : numpy.float64
            A 1d array with the moments.
    '''

    Moments = np.ones(n+1)
    for l in range(1, n+1):
        for k in range(len(WaveletCoef)):
            for m in range(l):
                Moments[l] += 1/((2**l - 1) * np.sqrt(2)) * (
                    WaveletCoef[k] * binom(l, m) * (k+1)**(l-m) * Moments[m])
    return Moments


def InnerProductPhiX(alpha, J, k, Moments):
    '''
    This function calculates the inner product between `x**alpha` and
    :math:`\phi_{J,k}`.

    INPUT:
        alpha : int
            The power of x
        J, k : int
            The indices for phi.
        Moments : numpy.float64
            A 1d array of moments for phi, up to power alpha. Can be
            calculated using the function moments().
    OUTPUT:
        i : numpy.float64
            The inner product.

    '''

    i = 0
    for l in range(alpha+1):
        i += 2**(-J+J/2-J*alpha) * binom(alpha, l) * k**(alpha-l) * Moments[l]
    return i


def BoundaryWavelets(phi, J, WaveletCoef, AL=None, AR=None):
    '''
    This function evaluates the left boundary functions.

    INPUT:
        phi : numpy.float64
            The scaling function at scale 0. (1d array)
        J : int
            The scale the scaling function has to have.
        WaveletCoef : numpy.float64
            The wavelet coefficients must sum to :math:`\sqrt{2}`.
            For Daubechies 2 they can be found using
            `np.flipud(pywt.Wavelet('db2').dec_lo)`.
        AL=None : numpy.float64
            The left orthonormalisation matrix, if this is not suplied
            the functions will not be orthonormalized.
        AR=None : numpy.float64
            The right orthonormalisation matrix, if this is not
            suplied the functions will not be orthonormalized.
    OUTPUT:
        xj : numpy.float64
            2d numpy array with the boundary funtions in the columns.

    '''

    a = int(len(WaveletCoef)/2)
    kLeft = np.arange(-2*a+2, 1)
    kRight = np.arange(2**J-2*a+1, 2**J)
    Moment = Moments(WaveletCoef, a-1)
    OneStep = len(phi)//(2*a-1)
    xj = np.zeros((OneStep, 2*a))
    PhiLeft = np.zeros((len(kLeft), OneStep))
    PhiRight = np.zeros((len(kRight), OneStep))
    for i in range(len(kLeft)):
        PhiLeft[i] = DownSample(phi, kLeft[i], OneStep, J)
        PhiRight[i] = DownSample(phi, kRight[i], OneStep, J)
    for b in range(a):
        for k in range(len(kLeft)):
            xj[:int((2*a-1+kLeft[k])*2**(-J)*OneStep), b] += (
                InnerProductPhiX(b, J, kLeft[k], Moment) *
                PhiLeft[k, :int((2*a-1+kLeft[k]) * 2**(-J) * OneStep)])
            xj[int(2**(-J)*kRight[k]*OneStep):, a+b] += (
                InnerProductPhiX(b, J, kRight[k], Moment) *
                PhiRight[k, int(2**(-J)*kRight[k]*OneStep):])
    if AL is None or AR is None:
        return xj
    else:
        x = np.zeros(np.shape(xj))
        for i in range(a):
            for j in range(a):
                x[:, i] += xj[:, j] * AL[i, j]
        for i in range(a):
            for j in range(a):
                x[:, i+a] += xj[:, j+a] * AR[i, j]
        return x
