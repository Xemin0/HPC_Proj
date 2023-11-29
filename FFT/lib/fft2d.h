/*
 * 2D FFT 
 *
 * 1. Apply 1D FFT along rows
 * 2. Apply 1D FFT along columns of the transformed data
 */

#include<complex>


typedef std::complex<double> Complex;


void fft_2d(Complex **img, int rows, int cols);

// !!!! MAY NEED AN OVERLOAD FOR FLATTENED 1D INPUT !!!! // 
//void fft_2d(Complex *img, int rows, int cols);
