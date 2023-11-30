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
    // fftw_complex to Complex dtype for 1D array 
    for (int i = 0; i < N; i++)
    {
        x[i] = Complex(arr[i][0], arr[i][1]);
    }
}

void Complex2fftw(Complex *x, fftw_complex *arr, unsigned int N)
{
    // Complex to fftw_complex dtype for 1D array
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

// 2D - FFT from FFTW 
// ##### for 2D input ##### //
void fftw_2d_wrapper(Complex **img, int rows, int cols)
{
    /*
     * Wrapper for 2D FFT call from FFTW library
     */
    // Need to be flattened 
    fftw_complex *mat_in, *mat_out;
    mat_in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * rows * cols);
    mat_out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * rows * cols);
 

    // Copy input into mat_in
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
        {
            mat_in[i * cols + j][0] = img[i][j].real();
            mat_in[i * cols + j][1] = img[i][j].imag();
        }

    // Creating plan for 2D FFT in FFTW
    fftw_plan p;
    p = fftw_plan_dft_2d(rows, cols, mat_in, mat_out,
                         FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p);

    // Copy the output back to input img
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            img[i][j] = Complex(mat_out[i * cols + j][0], mat_out[i * cols + j][1]);

    // clean up
    fftw_destroy_plan(p);
    fftw_free(mat_in);
    fftw_free(mat_out);
}

// !!!! MAY NEED AN OVERLOAD FOR FLATTENED 1D INPUT !!!! // 
// #### For flattened 1D row-major input #### //
void fftw_2d_wrapper(Complex *img, int rows, int cols)
{
    /*
     * Wrapper for 2D FFT call from FFTW library
     */

    fftw_complex *mat_in, *mat_out;
    // reinterpret the memory storage instead of allocating new memory
    mat_in = reinterpret_cast<fftw_complex*>(img); 
    mat_out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * rows * cols);

    // Creating plan for 2D FFT in FFTW
    fftw_plan p;
    p = fftw_plan_dft_2d(rows, cols, mat_in, mat_out,
                         FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p);

    // Convert to Complex dtype
    fftw2Complex(mat_out, img, rows * cols);

    // Clean up 
    fftw_destroy_plan(p);
    fftw_free(mat_out);
}
