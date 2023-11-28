/*
 * 1-D FFT with Iterative Cooley-Tukey Algorithm
 * 
 */

#include <iostream>
#include <complex>
#include <stdlib.h>
#include <cmath>
#include <fstream>

#include "./lib/fftw_wrapper.h" // FFTW method wrapper
#include "./lib/iterative_CT.h" // iterative FFT
#include "./lib/recursive_CT.h" // recursive FFt
#include "./lib/loader.h" // DataLoader
#include "./lib/vec_utils.h" // vector manipulations
#include "./lib/eval_correctness.h" // subroutines to verify correctness
#include "./lib/eval_performance.h" // subroutines to measure performance
#include "./lib/timer.h"	// timer
using namespace std;

typedef std::complex<double> Complex;


// Main Driver for Testing
int main()
{
	/*
    unsigned int N = 16;
    // randomize a vector
    // vec      : base vec
    // vec_fftw : used to store results from FFTW

    Complex *vec = rand_vec(N);
    //Complex *vec_fftw = rand_vec(N);
    cout << "Input vec:" << endl;
    show_vec(vec, N);

	cout << endl;

    // Creating a Copy of vec 
    // used by our method
    Complex *vec_fftw = (Complex*)malloc(N * sizeof(Complex));
    copy_vec(vec, vec_fftw, N);


    // ####### FFT standard
    fftw_1d_wrapper(vec_fftw, N);

    // ####### FFT CT Method
    fft_it_1d(vec, N);

    // Show both vecs
    cout << "FFT result by standard library:" << endl;
    show_vec(vec_fftw, N);

    cout << "FFT result by CT Method:" << endl;
    show_vec(vec, N);

    // Check if two vecs are equal 
    cout << boolalpha;
    cout << "Are FFT results the same?: " << bool(areVecsEqual(vec_fftw, vec, N)) << endl;
    
    // free memory
    free(vec);
    free(vec_fftw);
	*/


    // ******** Testing Given Data set ******* // 

	/*
	 * Available 1D FFT methods available for testing:
	 *
	 * 		- fft_it_1d 		: 1D FFT Iterative Method
	 *	 	- fft_re_1d			: 1D FFT Recursive Method
	 * 		- fftw_1d_wrapper	: 1D FFT method from FFTW
	 *		- 
	 */

    // Load Data
    Dataset1D finger1;

	//******** Validating the Correctness ********//


    // Eval the Correctness of Iterative 1D FFT and output to a file
    FFT1d_4Data(finger1, // Dataset
				fft_it_1d, // FFT method to test
                true, // if write toFile
                "our1d_it.txt");// filename
    /*
    // Eval the Correctness of FFTW's 1D FFT and output to a file
    FFT1d_4Data(finger1, // Dataset
                fftw_1d_wrapper, // FFT method to test
                true, // if write toFile
                "our1d_fftw.txt");// filename

    // Eval the Correctness of Recursive 1D FFT and output to a file
    FFT1d_4Data(finger1, // Dataset
				fft_re_1d, // FFT method to test
                true, // if write toFile
                "our1d_re.txt");// filename

	*/
	//*********************************************//


	//******** Evaluate the Performance ********//
	
	// Eval the Average Time Performing Iterative 1D FFT and output to a file
	eval_FFT1d_4Data(finger1,	// Dataset
					 fft_it_1d, // FFT method to test
					 2,			// warm up runs (excluded in eval)
					 5,			// testruns to take the average of
					 true,		// if write to file
					 "our1d_iter"); // base filename


	//*********************************************//
    //fftw_free(vec_in);
    //fftw_free(vec_out); // no need to reclaim memory for stack allocation
}
