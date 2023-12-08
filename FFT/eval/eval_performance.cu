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
                          // HighPrecisionTimer that measure both CPU and GPU time
#include "../lib/loader.h"
using namespace std;

typedef complex<double> Complex;

// Define the function pointer
typedef void (*FuncPtr)(Complex*, int);     // for 1D FFT methods
typedef void (*FuncPtr2)(Complex**, int, int);   // for 2D FFT methods


// ********** 1D FFT Performance Evaluation *********** //

float time_FFT1d_4Data(Dataset1D& ds, FuncPtr func, bool isCPU) // ## May need to return HighPrecisionTimer object
{
	/*
	 * Time a single run of provided 1D FFT method over the whole dataset
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


    float tot_time = 0.0;
    //unsigned long start, end;

    // high precision timer 
    HighPrecisionTimer timer;

    Complex tmp[truncated_cols];
    // 1D FFT for each channel/column
    for (int i = 0; i < rows; i++) 
        for (int k = 0; k < depth; k++){ 
            // load channel/row data to tmp
            for (int j = 0; j < truncated_cols; j++){
                tmp[j] = ds.getElement(i+1, j+1, k+1); // Copy by value ?? not by reference??
            }   

            // 1D FFT with provided method (as a function pointer)
			//start = get_time();
            timer.Start();
            func(tmp, truncated_cols);
			//end = get_time();
            timer.Stop();

			// Aggregate the timed result
			//tot_time += end - start;
            tot_time += timer.Elapsed(isCPU);
        }  
    return tot_time; 
}

float eval_FFT1d_4Data(Dataset1D& ds, FuncPtr func,
                      bool isCPU,
					  int warmup, int testruns,
					  bool toFile, std::string filename)
{
 	/*
     * Average Performance of 1D FFT for a given Dataset in microsecond (us)
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
    cout << "Writing average time of 1D FFT to: " << time_filename << endl;

	float avg_t = 0;

	// Warm up runs
	for (int i = 0; i < warmup; i++)
		time_FFT1d_4Data(ds, func);


	// Recording times and take the average
	for (int i = 0; i < testruns; i++)
		avg_t += time_FFT1d_4Data(ds, func, isCPU);

	avg_t /= testruns;

	// write to the file if specified
	if (toFile)      
		timeFile << "Average time of 1D FFT over the whole 1D-Dataset of " << testruns << " runs: " << avg_t << " us" << endl;

    timeFile.close();
	return avg_t;
}


// ********** 2D FFT Performance Evaluation *********** //

float time_FFT2d_4Data(Dataset2D& ds, FuncPtr2 func, bool isCPU)
{
    /*  
     * Time a single run of provided 2D FFT method over the whole dataset
     */
    int rows, cols, nImgs;   
    ds.getDimensions(rows, cols, nImgs);
    //cout << "rows = " << rows << endl;
    //cout << "cols = " << cols << endl;
    //cout << "nImgs = " << nImgs << endl;

    // Iterative FFT requires input size to be a power of 2
    int truncated_cols = log2(cols);
    truncated_cols = static_cast<int>(pow(2, truncated_cols));
    //cout << "resized column size: \t" << truncated_cols << endl;

    int truncated_rows = log2(rows);
    truncated_rows = static_cast<int>(pow(2, truncated_rows));
    //cout << "resized row size: \t" << truncated_rows << endl;


    float tot_time = 0;
    //unsigned long start, end;

    // high precision timer
    HighPrecisionTimer timer;

    // 2D FFT for each image
    for (int i = 0; i < 20; i++)  // **** Change current value back to nImgs
    {
        // Get image from the dataset by index
        Complex** curr_img = ds.getImage(i, false);

        // 2D FFT with provided method (as a function pointer)
        //start = get_time();
        timer.Start();
        func(curr_img, truncated_rows, truncated_cols);
        //end = get_time();
        timer.Stop();

        // clean up
        for (int j = 0; j < rows; j++)
            delete[] curr_img[j];
        delete[] curr_img;

        // Aggregate the timed result
        //tot_time += end - start;
        tot_time += timer.Elapsed(isCPU);
    }   
    return tot_time; 
}


float eval_FFT2d_4Data(Dataset2D& ds, FuncPtr2 func, 
                      bool isCPU,
                      int warmup, int testruns,
                      bool toFile, std::string filename)
{
    /*
     * Average Performance of 2D FFT for a given Dataset in microsecond (us)
     * (MAYBE) FLOPrate ** to be completed ** 
     *
     * Warmup runs are excluded in evaluation
     * 
     * - FuncPtr2: Function pointer of 2D FFT method
     *            in the form of `void func(Complex **x, int rows, int cols)` 
     */

    std::string root_path = "./Data/Results/";

    // File name preparation
    string time_filename = root_path + filename + "_us.dat";

    
    ofstream timeFile(time_filename); 
    if (!timeFile.is_open()){
        cerr << "Failed to open file for writing!" << endl;
    }
    cout << "Writing average time of 2D FFT to: " << time_filename << endl;

    float avg_t = 0;

    // Warm up runs
    for (int i = 0; i < warmup; i++)
        time_FFT2d_4Data(ds, func);


    // Recording times and take the average
    for (int i = 0; i < testruns; i++)
        avg_t += time_FFT2d_4Data(ds, func, isCPU);

    avg_t /= testruns;

    // write to the file if specified
    if (toFile)      
        timeFile << "Average time of 1D FFT over the whole 1D-Dataset of " << testruns << " runs: " << avg_t << " us" << endl;

    timeFile.close();
    return avg_t;
}

