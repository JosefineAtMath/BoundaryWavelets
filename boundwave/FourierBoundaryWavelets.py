# -*- coding: utf-8 -*-
"""
This module is used for calculations of the boundary wavelets in the frequency
domain.

The BoundaryWavelets.py package is licensed under the MIT "Expat" license.

Copyright (c) 2019: Josefine Holm and Steffen L. Nielsen.
"""
# =============================================================================
# Imports
# =============================================================================
import numpy as np
import boundwave.BoundaryWavelets as BW


# =============================================================================
# Functions
# =============================================================================
def Rectangle(Scheme):
    '''
    The Fourier transform of a rectangular window function.

    INPUT:
        Scheme : numpy.float64
            A numpy array with the frequencies in which to sample.
    OUTPUT:
        chi : numpy.complex128
            A numpy array with the window function sampled in the
            freqency domain.

    '''

    chi = np.zeros(len(Scheme), dtype=np.complex128)
    for i in range(len(Scheme)):
        if Scheme[i] == 0:
            chi[i] = 1
        else:
            chi[i] = (1-np.exp(-2*np.pi*1j*Scheme[i]))/(2*np.pi*1j*Scheme[i])
    return chi


def ScalingFunctionFourier(WaveletCoef, J, k, Scheme, Win, P=20):
    r'''
    This function evaluates the Fourier transform of the scaling function,
    :math:`\phi_{j,k}`, sampled in scheme.

    INPUT:
        WaveletCoef : numpy.float64
            The wavelet coefficients, must sum to :math:`\sqrt{2}`.
            For Daubechies 2 they can be found using
            `np.flipud(pywt.Wavelet('db2').dec_lo)`.
        J : int
            The scale.
        k : int
            The translation.
        Scheme : numpy.float64
            The points in which to evaluate.
        Window=Rectangle : numpy.complex128
            The window to use on the boundary functions.
        P=20 : int
            The number of factors to include in the infinite product
            in the Fourier transform of phi.
    OUTPUT:
        phi : numpy.complex128
            :math:`\hat{\phi}_{j,k}`

    '''

    h = WaveletCoef*np.sqrt(2)/2
    e = (Scheme[-1]-Scheme[0])/len(Scheme)
    phi = np.zeros((len(Scheme), P, len(h)), dtype=complex)
    for i in range(P):
        for l in range(len(h)):
            phi[:, i, l] = h[l]*np.exp(-2*np.pi*1j*l*2**(-i-J-1)*Scheme)
    phi = np.sum(phi, axis=2, dtype=np.complex128)
    phi = (2**(-J/2) * np.exp(-2*np.pi*1j*k*2**(-J)*Scheme) *
           np.prod(phi, axis=1, dtype=np.complex128))
    PhiAstChi = np.convolve(phi, Win, mode='same') * e
    return PhiAstChi


def FourierBoundaryWavelets(J, Scheme, WaveletCoef, AL=None, AR=None,
                            Win=Rectangle):
    r'''
    This function evaluates the Fourier transformed boundary functions
    for db2.

    INPUT:
        J : int
            The scale.
        Scheme : numpy.float64
            The sampling scheme in the Fourier domain.
        WaveletCoef : numpy.float64
            The wavelet coefficients, must sum to :math:`\sqrt{2}`.
            For Daubeshies 2 they can be found using
            `np.flipud(pywt.Wavelet('db2').dec_lo)`.
        AL=None : numpy.float64
            The left orthonormalisation matrix, if this is not
            supplied the functions will not be orthonormalized. Can be
            computed using
            :py:func:`boundwave.Orthonormal.OrthoMatrix`.
        AR=None : numpy.float64
            The right orthonormalisation matrix, if this is not
            supplied the functions will not be orthonormalized. Can be
            computed using
            :py:func:`boundwave.Orthonormal.OrthoMatrix`.
        Win= :py:func:`Rectangle` : numpy.complex128
            The window to use on the boundary functions.
    OUTPUT:
        x : numpy.complex128
            2d numpy array with the boundary functions in the columns;
            orthonormalised if `AL` and `AR` given.

    '''

    a = int(len(WaveletCoef)/2)
    kLeft = np.arange(-2*a+2, 1)
    kRight = np.arange(2**J-2*a+1, 2**J)
    xj = np.zeros((len(Scheme), 2*a), dtype=complex)
    Moment = BW.Moments(WaveletCoef, a-1)
    FourierPhiLeft = np.zeros((len(kLeft), len(Scheme)), dtype=complex)
    FourierPhiRight = np.zeros((len(kRight), len(Scheme)), dtype=complex)
    Window = Win(Scheme)
    for i in range(len(kLeft)):
        FourierPhiLeft[i] = ScalingFunctionFourier(WaveletCoef, J,
                                                   kLeft[i], Scheme, Window)
        FourierPhiRight[i] = ScalingFunctionFourier(WaveletCoef, J,
                                                    kRight[i], Scheme, Window)
    for b in range(a):
        xj[:, b] = np.sum(np.multiply(BW.InnerProductPhiX(
            b, J, kLeft, Moment), np.transpose(FourierPhiLeft)), axis=1)
        xj[:, b+a] = np.sum(np.multiply(BW.InnerProductPhiX(
            b, J, kRight, Moment), np.transpose(FourierPhiRight)), axis=1)
    if type(AL) is None or type(AR) is None:
        return xj
    else:
        x = np.zeros(np.shape(xj), dtype=complex)
        for i in range(a):
            for j in range(a):
                x[:, i] += xj[:, j] * AL[i, j]
        for i in range(a):
            for j in range(a):
                x[:, i+a] += xj[:, j+a] * AR[i, j]
        return x
