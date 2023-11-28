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

//#include "../lib/iterative_CT.h"
//#include "../lib/fftw_wrapper.h"
#include "../lib/loader.h"
using namespace std;

typedef complex<double> Complex;

// Define the function pointer
typedef void (*FuncPtr)(Complex*, int);

void FFT1d_4Data(Dataset1D& ds, FuncPtr func, 
				 bool toFile, std::string filename)
{
 	/*
     * Correctness of FFT for a given Dataset 
     * 
     * - FuncPtr: Function pointer of FFT method
	 * 			  in the form of `void func(Complex *x, int N)` 
     */

	std::string root_path = "./Data/Results/";

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
    filename = root_path + filename;

    
    ofstream fftFile(filename); 
    if (!fftFile.is_open()){
        cerr << "Failed to open file for writing!" << endl;
    }
    cout << "Writing results to: " << filename << endl;


    // get data for each channel/column
    Complex *tmp = new Complex[truncated_cols];

    // FFT for each channel/column
    for (int i = 0; i < 1; i++) // ***  change 1 to rows
        for (int k = 0; k < 1; k++){ // *** change 1 to depth
            // load channel/row data to tmp
            for (int j = 0; j < truncated_cols; j++){
                tmp[j] = ds.getElement(i+1, j+1, k+1); // Copy by value ?? not by reference??
            }

            // FFT with provided method (as a function pointer)
            func(tmp, truncated_cols);

            // Write Our result to the output array 
            // Column Major
            for (int j = 0; j < truncated_cols; j++)
                ds.setElement(tmp[j], i+1, j+1, k+1, true); // true: set element for ds.fft_data
            
            // store the result for current channel
            if (toFile)      
                for (int j = 0; j < truncated_cols; j++){
                    fftFile << "Channel " << i << ", FFT[" << j << "] = " << tmp[j] << endl;
                }
        }

    fftFile.close();
    delete[] tmp;
}
