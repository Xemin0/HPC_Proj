/*
 * Wrapper for FFT methods from FFTW library
 * along with utilities for dtype conversion
 * 
 * ### Requires linker and include path of FFTW while linking after compiling ###
 * ### See `makefile` for details ###
 */

#ifndef LIB_FFTW_WRAPPER_H_
#define LIB_FFTW_WRAPPER_H_

#include <fftw3.h>

typedef std::complex<double> Complex;

/* ### DTYPE conversion utils // In essence, 'COPY' methods
 *
 * `reinterpret_cast<Complex*>` or 
 * `reinterpret_cast<fftw_complex*>` recommended
 */
void fftw2Complex(fftw_complex *arr, Complex *x, unsigned int N);
// for 1D array

void Complex2fftw(Complex *x, fftw_complex *arr, unsigned int N);
// for 1D array

// ### FFTW Methods Wrappers

// 1D - FFT from FFTW
void fftw_1d_wrapper(Complex *x, int N);

// 2D - FFT from FFTW
// #### For 2D input #### //
void fftw_2d_wrapper(Complex **img, int rows, int cols);

// !!!! MAY NEED AN OVERLOAD FOR FLATTENED 1D INPUT !!!! //
// 2D - FFT from FFTW
// #### For flattened 1D row-major input #### //
void fftw_2d_wrapper(Complex *img, int rows, int cols);

#endif /* LIB_FFTW_WRAPPER_H_ */
