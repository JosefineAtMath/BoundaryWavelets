# BoundaryWavelets
Wavelets are widely used for signal analysis and compression. More often than not the signal in question is only known on a bounded interval. Therefore, this package implements an orthonormal wavelet basis -- including special boundary wavelets -- for L^2([0,1]), as well as the Fourier transform of the basis.

Classical approaches to the boundary problem involves extending the signal artificially beyond the interval in which it is known. We find and demonstrate through numerical experiments that the proposed boundary wavelet decomposition is better on average compared to such classical approaches.
