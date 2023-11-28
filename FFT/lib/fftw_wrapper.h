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

void Complex2fftw(Complex *x, fftw_complex *arr, unsigned int N);

// ### FFTW Methods Wrappers

// 1D - FFT from FFTW
void fftw_1d_wrapper(Complex *x, int N);


#endif /* LIB_FFTW_WRAPPER_H_ */
