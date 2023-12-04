/*
 * Header for Iterative 1-D FFT with OpenMP
 */

#ifndef FFT1D_OPENMP_H_
#define FFT1D_OPENMP_H_

#include <complex>

typedef std::complex<double> Complex;

// Iterative 1-D FFT with OpenMP
void fft_it_1d_openMP(Complex *x, int N);

#endif /* FFT1D_OPENMP_H_ */

