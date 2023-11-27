/*
 * Subroutines to Verify the Correctness of FFT methods
 *
 * ## Default Output Path will be in `./Data/Results/`
 */


#include <iostream>
#include <complex>
#include <cmath>
#include <fstream>
#include <stdlib.h>

#include "../lib/iterative_CT.h"
#include "../lib/fftw_wrapper.h"
#include "../lib/loader.h"
using namespace std;

typedef complex<double> Complex;

void FFT1d_4Data(Dataset& ds, bool ifIter = true, bool toFile = true, std::string root_path = "./Data/Results/")
{
 	/*
     * Performance of FFT for a given Dataset in microsecond (us)
     * 
     * bool ifIter: whether to use Iterative Cooley-Tukey or the Recursive
     * 
     */
    int rows, cols, depth;   
    ds.getDimensions(rows, cols, depth);
    cout << "rows = " << rows << endl;
    cout << "cols = " << cols << endl;
    cout << "depth = " << depth << endl;

    // Iterative FFT requires input size to be a power of 2
    int truncated_cols = log2(cols);
    truncated_cols = pow(2, truncated_cols);
    cout << "resized column size: \t" << truncated_cols << endl;

    // File name preparation
    string ourfile;
    if (ifIter)
    {
       ourfile = root_path + "iter_results_our.txt";
    }
    else
    {
       ourfile = root_path + "recur_results_our.txt";
    }
    string fftwfile = root_path + "fftw_results.txt";
    
    ofstream fftOur(ourfile); // for our itermethod
    ofstream fftFFTW(fftwfile); // for fftw result
    if (!fftOur.is_open() || !fftFFTW.is_open()){
        cerr << "Failed to open file for writing!" << endl;
    }
    cout << "Writing results to: " << ourfile << endl;
    cout << "Writing FFTW results to: " << fftwfile << endl;



    // get data for each channel/column
    Complex *tmp = new Complex[truncated_cols];
    Complex *tmp_fftw = new Complex[truncated_cols];

    // FFT for each channel/column
    for (int i = 0; i < 1; i++) // ***  change 1 to rows
        for (int k = 0; k < 1; k++){ // *** change 1 to depth
            // load channel/row data to tmp
            for (int j = 0; j < truncated_cols; j++){
                tmp[j] = ds.getElement(i+1, j+1, k+1); // Copy by value ?? not by reference??
                tmp_fftw[j] = tmp[j];
            }

            // FFT with our method
            fft_it_1d(tmp, truncated_cols);

            // FFT with FFTW Library 
            fftw_1d_wrapper(tmp_fftw, truncated_cols);

            // Write Our result to the output array 
            // Column Major
            for (int j = 0; j < truncated_cols; j++)
                ds.fft_data[k * rows * truncated_cols + j * rows + i] = tmp[j];
            
            // store the result for current channel
            if (toFile)      
                for (int j = 0; j < truncated_cols; j++){
                    fftOur << "Channel " << i << ", FFT[" << j << "] = " << tmp[j] << endl;
                    fftFFTW << "Channel " << i << ", FFT[" << j << "] = " << tmp_fftw[j] << endl;

                }
        }

    fftOur.close();
    fftFFTW.close();
    delete[] tmp;
    delete[] tmp_fftw;
}
