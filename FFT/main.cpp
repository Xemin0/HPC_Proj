/*
 * 1-D FFT with Iterative Cooley-Tukey Algorithm
 * 
 */

#include <iostream>
#include <complex>
#include <stdlib.h>
#include <cmath>
#include <fstream>

#include "./lib/fftw_wrapper.h"
#include "./lib/iterative_CT.h"
#include "./lib/loader.h" // DataLoader
#include "./lib/vec_utils.h"
#include "./lib/eval_correctness.h"
#include "./lib/timer.h"
using namespace std;

typedef std::complex<double> Complex;


// Main Driver for Testing
int main()
{
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


    // ******** Testing Given Data set ******* // 

	/*
	 * Available 1D FFT methods available for testing:
	 *
	 * 		- fft_it_1d 		: 1D FFT Iterative Method
	 * 		- fftw_1d_wrapper	: 1D FFT method from FFTW
	 *		- 
	 */

    // Load Data
    Dataset finger1;

    // Eval the Correctness of Iterative 1D FFT and output to a file
    FFT1d_4Data(finger1, // Dataset
				fft_it_1d, // FFT method to test
                true, // if write toFile
                "our1d_it.txt");// filename
    
    FFT1d_4Data(finger1, // Dataset
				fftw_1d_wrapper, // FFT method to test
                true, // if write toFile
                "our1d_fftw.txt");// filename
    //fftw_free(vec_in);
    //fftw_free(vec_out); // no need to reclaim memory for stack allocation
}
