# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 11:38:58 2018

@author: josefine

The BoundaryWavelets.py package is licensed under the MIT "Expat" 
Copyright (c) 2018: Josefine Holm and Steffen L. Nielsen.
"""
# =============================================================================
# Imports
# =============================================================================
import numpy as np
import pywt
import matplotlib.pyplot as plt
import BoundaryWavelets as BW
import FourierBoundaryWavelets as FBW
import Orthonormal as Ot
# =============================================================================
# Functions
# =============================================================================

def GeneralTest(Wavelet='db2',J=2,epsilon=1/7,Length=1792,TimeOnly=False):
    '''
    A test function used in the paper. PLots the boundary wavelets in time and 
    compare them with the boundary wavelets created in frequency by using an 
    inverse Fourier transform. \n
    INPUT:
        Wavelet='db2':
            str: The wavelet to use in the test.
        J=2:
            int: The scale. If J is too small for the scaling function to be supported
            within the interval [0,1] it is changed to the smallest possible J.
        epsilon=1/7:
            float: The sampling dencity in the frequency domain, i.e. the distance between
            samples. In Generalized sampling this must be at least 1/7 for db2.
        Length=1792:
            int: The number of samples in the frequency domain. The standard is chosen
            because 1792/7=256.
        TimeOnly=False:
            bool: If TImeOnly=True the boundary functions are only constructed in 
            time, not in frequency.
    '''
    WaveCoef = np.flipud(pywt.Wavelet(Wavelet).dec_lo)
    phi      = pywt.Wavelet(Wavelet).wavefun(level=15)[0][1:]
    a        = int(len(WaveCoef)/2)
    OneStep  = len(phi)//(2*a-1)
    if J<a:
        J = a
        print('J has been changed to',J)
    AL,AR    = Ot.OrthoMatrix(J,WaveCoef,phi)
    phiNorm  = np.sqrt(1/OneStep*np.sum(np.abs(BW.DownSample(phi,0,OneStep,J)**2)))
    BoundaryT= BW.BoundaryWavelets(phi/phiNorm,J,WaveCoef,AL=AL,AR=AR)
    NormT    = np.zeros(2*a)
    for i in range(2*a):
        NormT[i] = np.sqrt(1/OneStep*np.sum(np.abs(BoundaryT[:,i])**2))
    IP = 0
    for i in range(a):
        IP += i
    LeftInnerProdT = np.zeros(IP)
    ij = 0
    for i in range(a):
        for j in range(i+1,a):
            LeftInnerProdT[ij] = 1/OneStep*np.sum(BoundaryT[:,i]*BoundaryT[:,j])
            ij += 1
    RightInnerProdT = np.zeros(IP)
    ij = 0
    for i in range(a):
        for j in range(i+1,a):
            RightInnerProdT[ij] = 1/OneStep*np.sum(BoundaryT[:,i+a]*BoundaryT[:,j+a])
            ij += 1
    if TimeOnly==True:
        print('Norms of functions in time', NormT)
        print('Inner products in time',LeftInnerProdT,RightInnerProdT)
        plt.figure()
        for i in range(a):
            plt.plot(np.linspace(0,1,len(BoundaryT[:,i])),BoundaryT[:,i],label='$\phi^L_{%d,%d}$'%(J,i))
        for i in range(a):
            plt.plot(np.linspace(0,1,len(BoundaryT[:,i+a])),BoundaryT[:,i+a],label='$\phi^R_{%d,%d}$'%(J,i))
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
    else:
        s         = int(Length*epsilon)
        d         = 1/epsilon
        Scheme    = np.linspace(-Length*epsilon/2,Length*epsilon/2,Length,endpoint=False)
        BoundaryF = FBW.FourierBoundaryWavelets(J,Scheme,WaveCoef,AL=AL,AR=AR)
        NormF     = np.zeros(2*a)
        for i in range(2*a):
            NormF[i]   = np.sqrt(epsilon*np.sum(np.abs(BoundaryF[:,i])**2))
        LeftInnerProdF = np.zeros(IP,dtype=complex)
        ij    = 0
        for i in range(a):
            for j in range(i+1,a):
                LeftInnerProdF[ij] = epsilon*np.sum(BoundaryF[:,i]*BoundaryF[:,j])
                ij += 1
        RightInnerProdF = np.zeros(IP,dtype=complex)
        ij = 0
        for i in range(a):
            for j in range(i+1,a):
                RightInnerProdF[ij] = epsilon*np.sum(BoundaryF[:,i+a]*BoundaryF[:,j+a])
                ij += 1
        print('Norms of functions in time', NormT)
        print('Norms of functions i frequency',NormF)
        print('Inner products in time',LeftInnerProdT,RightInnerProdT)
        print('Inner products in frequency',LeftInnerProdF,RightInnerProdF)
        InverseBoundaryF=np.zeros((s,2*a),dtype=complex)
        for i in range(2*a):
            InverseBoundaryF[:,i] = d**(1/2)*np.fft.ifft(d**(-1/2)*np.concatenate((BoundaryF[len(BoundaryF[:,i])//2:,i],BoundaryF[:len(BoundaryF[:,i])//2,i])),norm='ortho')[:s]
            Fnorm=np.sqrt(1/256*np.sum(np.abs(InverseBoundaryF[:,i])**2))
            InverseBoundaryF[:,i]/=Fnorm
        plt.figure()
        plt.plot(np.linspace(0,1,len(BoundaryT[:,0])),BoundaryT[:,0],label='Scaling functions in time',color='C0')
        plt.plot(np.linspace(0,1,s),np.real(InverseBoundaryF[:,0]),label='Scaling functions in frequency',color='C1')
        for i in range(1,a):
            plt.plot(np.linspace(0,1,len(BoundaryT[:,0])),BoundaryT[:,i],color='C0')
            plt.plot(np.linspace(0,1,s),np.real(InverseBoundaryF[:,i]),color='C1')
        for i in range(a):
            plt.plot(np.linspace(0,1,len(BoundaryT[:,0])),BoundaryT[:,i+a],color='C0')
            plt.plot(np.linspace(0,1,s),np.real(InverseBoundaryF[:,i+a]),color='C1')
        plt.legend(bbox_to_anchor=(0,0.95,1,0.2), loc="lower left",ncol=1,mode='expand',framealpha=1)
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
    return 

GeneralTest(Wavelet='db2',J=2,TimeOnly=False)
