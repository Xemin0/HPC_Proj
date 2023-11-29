/*
 * 2D FFT 
 *
 * 1. Apply 1D FFT along rows
 * 2. Apply 1D FFT along columns of the transformed data
 */

#include<complex>

#include "../lib/iterative_CT.h" // void fft_it_1d(Complex *x, int N);

typedef std::complex<double> Complex;


void fft_2d(Complex **img, int rows, int cols)
{
    /*
     * 2D FFT
     *
     *** it writes the result back to the input matrix ***
     * - rows: Height
     * - cols: Width
     */
    // Apply 1D FFT to each row
    for (int i = 0; i < rows; i++)
        fft_it_1d(img[i], cols);

    // Create a temp array to store each column
    Complex tmp[rows];
    for (int j = 0; j < cols; j++)
    {
        // copy each column into the tmp array for contiguous access of the entries
        for (int i = 0; i < rows; i++)
            tmp[i] = img[i][j];

        // Apply 1D FFT to the current column
        fft_it_1d(tmp, rows);
    
        // write the transformed data back to each column
        for (int i = 0; i< rows; i++)
            img[i][j] = tmp[i];
    }
}
