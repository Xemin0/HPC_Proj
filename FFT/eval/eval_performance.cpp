/*
 * Subroutines to Time and Calculate the Performance of FFT methods
 *
 *
 * - time evaluation
 * - FLOPrate evaluation
 *
 * ## Default Output Path will be in `./Data/Results/`, as a `.dat` file
 */


#include <iostream>
#include <complex>
#include <cmath>
#include <fstream>
#include <stdlib.h>

//#include "../lib/iterative_CT.h"
//#include "../lib/fftw_wrapper.h"

#include "../lib/timer.h" // get_time() return time in microsecond (us)
#include "../lib/loader.h"
using namespace std;

typedef complex<double> Complex;

// Define the function pointer
typedef void (*FuncPtr)(Complex*, int);

unsigned long time_FFT1d_4Data(Dataset1D& ds, FuncPtr func)
{
	/*
	 * Time a single run of provided FFT method over the whole dataset
	 */
	int rows, cols, depth;   
    ds.getDimensions(rows, cols, depth);
    //cout << "rows = " << rows << endl;
    //cout << "cols = " << cols << endl;
    //cout << "depth = " << depth << endl;

    // Iterative FFT requires input size to be a power of 2
    int truncated_cols = log2(cols);
    truncated_cols = static_cast<int>(pow(2, truncated_cols));
    //cout << "resized column size: \t" << truncated_cols << endl;


	unsigned long tot_time = 0, start, end;

	Complex tmp[truncated_cols];
    // FFT for each channel/column
    for (int i = 0; i < rows; i++) 
        for (int k = 0; k < depth; k++){ 
            // load channel/row data to tmp
            for (int j = 0; j < truncated_cols; j++){
                tmp[j] = ds.getElement(i+1, j+1, k+1); // Copy by value ?? not by reference??
            }   

            // FFT with provided method (as a function pointer)
			start = get_time();
            func(tmp, truncated_cols);
			end = get_time();

			// Aggregate the timed result
			tot_time += end - start;
        }  
	return tot_time; 
}

unsigned long eval_FFT1d_4Data(Dataset1D& ds, FuncPtr func, 
					  int warmup, int testruns,
					  bool toFile, std::string filename)
{
 	/*
     * Average Performance of FFT for a given Dataset in microsecond (us)
	 * (MAYBE) FLOPrate ** to be completed ** 
     *
	 * Warmup runs are excluded in evaluation
     * 
     * - FuncPtr: Function pointer of FFT method
	 * 			  in the form of `void func(Complex *x, int N)` 
     */

	std::string root_path = "./Data/Results/";

    // File name preparation
    string time_filename = root_path + filename + "_us.dat";

    
    ofstream timeFile(time_filename); 
    if (!timeFile.is_open()){
        cerr << "Failed to open file for writing!" << endl;
    }
    cout << "Writing average time to: " << time_filename << endl;

	unsigned long avg_t = 0;

	// Warm up runs
	for (int i = 0; i < warmup; i++)
		time_FFT1d_4Data(ds, func);


	// Recording times and take the average
	for (int i = 0; i < testruns; i++)
		avg_t += time_FFT1d_4Data(ds, func);

	avg_t /= testruns;

	// write to the file if specified
	if (toFile)      
		timeFile << "Average time of " << testruns << " runs: " << avg_t << " us" << endl;

    timeFile.close();
	return avg_t;
}
