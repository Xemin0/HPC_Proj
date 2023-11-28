/*
 * Recursive Method of Cooley-Tukey FFT Algorithm
 * 
 */

#ifndef LIB_RECURSIVE_CT_H_
#define LIB_RECURSIVE_CT_H_

#include <complex>

typedef std::complex<double> Complex;

// Recursive 1-D FFT
void fft_re_1d(Complex *x, int N);

#endif /* LIB_RECURSIVE_CT_H_ */
