# -*- coding: utf-8 -*-
"""
This module is used for testing the boundary wavelets on ECG data.

The BoundaryWavelets.py package is licensed under the MIT "Expat" license.

Copyright (c) 2019: Josefine Holm and Steffen L. Nielsen.
"""

import scipy.io as sp
import numpy as np
import pywt
import matplotlib.pyplot as plt
import ReconFunctions as RF


def TestPlot(Name='Data', Row=1, Section=214, J=7, N=12, Wavelet='db3'):
    '''
    This function makes decompositions and reconstructions of a chosen
    section of the data, both with boundary wavelets and with mirrored
    extension. The difference between the orignal signal and the two
    reconstructions are calculated and printed and all three signals
    are plotted in the same figure.

    INPUT:
        Name : str
            The MATLAB data file from which to load.
        Row : int
            The row in the dataset to use.
        Section : int
            Which section of the data to use. The samples that will be
            used are: `[Section*2**N:Section*2**N+2**N]`.
        J : int
            The scale.
        N : int
            The number of iterations to use in the cascade algorithm.
        Wavelet : str
            The name of the wavelet to be used. eg: `'db2'`.
    '''

    data = sp.loadmat(Name)
    phi = pywt.Wavelet(Wavelet).wavefun(level=14)[0][1:]
    phi = phi[::2**(14-N)]
    Signal = data['val'][Row, Section*2**N:Section*2**N+2**N]
    x1 = RF.DecomBoundary(Signal, J, Wavelet, phi, N=N)
    print(len(x1))
    NewSignal1 = np.real(RF.ReconBoundary(x1, J, Wavelet, phi, N=N))
    x2 = RF.DecomMirror(Signal, J, Wavelet, phi, N=N)
    NewSignal2 = np.real(RF.ReconMirror(x2, J, Wavelet, phi, N=N))
    dif1 = np.sum(np.abs(Signal-NewSignal1)**2)**(1/2)/2**N
    dif2 = np.sum(np.abs(Signal-NewSignal2)**2)**(1/2)/2**N
    print(dif1, dif2)

    plt.figure()
    plt.plot(Signal, label='Original')
    plt.plot(NewSignal1, label='Boundary wavelets')
    plt.plot(NewSignal2, label='Mirror')
    plt.xlabel('Sample index')
    plt.legend()
    return


def Test(Name='Data', Row=1, J=7, N=12, Wavelet='db3'):
    '''
    This function makes decompositions and reconstructions of several
    sections of the data, both with boundary wavelets and with
    mirrored extension. The differences between the orignal signal and
    the two reconstructions are calculated. The test is run for as
    many disjoint sections of the signal as possible.

    INPUT:
        Name : str
            The MATLAB data file from whichto load.
        Row : int
            The row in the dataset to use.
        J : int
            The scale.
        N : int
            The number of iterations to use in the cascade algorithm.
        Wavelet : str
            The name of the wavelet to be used. eg: `'db2'`.
    OUTPUT:
        Result : float64
            2D array. The first row is the difference between the
            original signal and the reconstruction using boundary
            wavelet. The second row is the difference between the
            original signal and the reconstruction using mirrored
            extension. The third row is the first row minus the second
            row. There is one collumn for each section of the signal.
    '''

    data = sp.loadmat(Name)
    phi = pywt.Wavelet(Wavelet).wavefun(level=14)[0][1:]
    phi = phi[::2**(14-N)]
    n = 0
    tests = int(len(data['val'][Row])/2**N)
    Result = np.zeros((3, tests))
    for i in range(tests):
        Signal = data['val'][Row, n:n+2**N]
        x1 = RF.DecomBoundary(Signal, J, Wavelet, phi, N=N)
        x2 = RF.DecomMirror(Signal, J, Wavelet, phi, N=N)
        NewSignal1 = np.real(RF.ReconBoundary(x1, J, Wavelet, phi, N=N))
        NewSignal2 = np.real(RF.ReconMirror(x2, J, Wavelet, phi, N=N))
        Result[0, i] = np.sum(np.abs(Signal-NewSignal1)**2)**(1/2)/2**N
        Result[1, i] = np.sum(np.abs(Signal-NewSignal2)**2)**(1/2)/2**N
        n += 2**N
    Result[2] = Result[0]-Result[1]
    plt.figure()
    plt.plot(Result[1], label='Mirror', color='C1')
    plt.plot(Result[0], label='Boundary', color='C0')
    plt.xlabel('Test signal')
    plt.ylabel('Difference')
    plt.legend()
    return Result

if __name__ == '__main__':
    TestPlot()
    Test = Test()
