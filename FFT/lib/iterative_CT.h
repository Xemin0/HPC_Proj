/*
 * Iterative Method of Cooley-Tukey FFT Algorithm
 * 
 */

#ifndef LIB_ITERATIVE_CT_H_
#define LIB_ITERATIVE_CT_H_

#include <complex>

typedef std::complex<double> Complex;

// bit reverse
void bitReverse(Complex *x, int N);

// Iterative 1-D FFT
void fft_it_1d(Complex *x, int N);

#endif /* LIB_ITERATIVE_CT_H_ */
