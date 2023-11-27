/*
 * Wrapper for FFT methods from FFTW library
 * along with utilities for dtype conversion
 * 
 * ### Requires linker and include path of FFTW while linking after compiling ###
 * ### See `makefile` for details ###
 */

#include <complex>
#include <fftw3.h>

typedef std::complex<double> Complex;

/* ### DTYPE conversion utils // In essence, 'COPY' methods
 *
 * `reinterpret_cast<Complex*>` or 
 * `reinterpret_cast<fftw_complex*>` recommended
 */
void fftw2Complex(fftw_complex *arr, Complex *x, unsigned int N)
{
    // fftw_complex to Complex dtype
    for (int i = 0; i < N; i++)
    {
        x[i] = Complex(arr[i][0], arr[i][1]);
    }
}

void Complex2fftw(Complex *x, fftw_complex *arr, unsigned int N)
{
    // Complex to fftw_complex dtype
    for (int i = 0; i < N; i++)
    {
        arr[i][0] = x[i].real();
        arr[i][1] = x[i].imag();
    }
}

// ### FFTW Methods Wrappers

// 1D - FFT from FFTW
void fftw_1d_wrapper(Complex *x, int N)
{
    /*
     * Wrapper for 1D FFT call from FFTW library
     */
    fftw_complex *vec_in, vec_out[N];
    // Reinterpret the memory storage instead of allocating new memory 
    vec_in = reinterpret_cast<fftw_complex*>(x);

    // Creating plan for 1D FFT in FFTW
    fftw_plan p;
    p = fftw_plan_dft_1d(N, vec_in, vec_out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p);
    
    // convert to Complex dtype
    fftw2Complex(vec_out, x, N);

    fftw_destroy_plan(p);
}

