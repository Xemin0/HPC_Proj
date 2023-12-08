/*
 * Main Driver to Test FFT Methods 
 */

#include <iostream>
#include <complex>
#include <stdlib.h>
#include <cmath>
#include <fstream>

#include "./lib/fftw_wrapper.h" // FFTW method wrapper
#include "./lib/iterative_CT.h" // iterative FFT
#include "./lib/recursive_CT.h" // recursive FFt
#include "./lib/fft2d.h"        // 2D FFT
#include "./lib/fft1d_cuda.h"   // 1D FFT w. CUDA

#include "./lib/loader.h"       // DataLoader
#include "./lib/eval_correctness.h" // subroutines to verify correctness
#include "./lib/eval_performance.h" // subroutines to measure performance
//#include "./lib/timer.h"	    // timer
#include "./lib/vec_utils.h"    // vector manipulations
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
    /*
    int N = 8;
    Complex *vec = rand_vec(N);
    cout << "Input vec:" << endl;
    show_vec(vec, N);

    Complex *vec_it = (Complex*)malloc(N * sizeof(Complex));
    copy_vec(vec, vec_it, N);
    cout << "copied raw vec:" << endl;
    show_vec(vec_it, N);

    fft1d_cu(vec, N);
    cout << "fft1d cuda:" << endl;
    show_vec(vec, N);

    cout << "===================" << endl;

    fft_it_1d(vec_it, N);
    cout << "it method:" << endl;
    show_vec(vec_it, N);
    
    cout << "end of individual test" << endl;
    */
    // ******** Testing Given DataSet ******* // 

	/*
	 * Available 1D FFT methods available for testing:
	 *
	 * 		- fft_it_1d 		: 1D FFT Iterative Method
	 *	 	- fft_re_1d			: 1D FFT Recursive Method
	 * 		- fftw_1d_wrapper	: 1D FFT method from FFTW
	 *		- fft1d_cu          : 1D FFT method w. CUDA
     *
     *
	 * Available 2D FFT methods available for testing:
     * 
     *      - fft2d             : 2D FFT
     *      - fftw_2d_wrapper   : 2D FFT method from FFTW
     *      - 
	 */

    // Load Data
    Dataset1D finger1;
    Dataset2D cifar10;

	//******** Validating the Correctness of 1D ********//

    // ************* 1D FFT *************** //

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
    */
    // Eval the Correctness of Recursive 1D FFT and output to a file
    //FFT1d_4Data(finger1, // Dataset
	//			fft_re_1d, // FFT method to test
    //            true, // if write toFile
    //            "our1d_re.txt");// filename
	
    FFT1d_4Data(finger1,
                fft1d_cu,
                true,
                "our1d_cu.txt");

    // ************* 2D FFT *************** //
    //FFT2d_4Data(cifar10,
    //            fftw_2d_wrapper,
    //            true,
    //            "out2d_fftw.txt");

    //FFT2d_4Data(cifar10,
    //            fft_2d,
    //            true,
    //            "out2d_custom.txt");

	//*********************************************************//


	//******** Evaluate the Performance ********//
	
    // ************* 1D FFT *************** //

	// Eval the Average Time Performing Iterative 1D FFT and output to a file
	// eval_FFT1d_4Data(finger1,	// Dataset
	// 				 fft_it_1d, // FFT method to test
	// 				 2,			// warm up runs (excluded in eval)
	// 				 5,			// testruns to take the average of
	// 				 true,		// if write to file
	// 				 "our1d_iter"); // base filename


    // ************* 2D FFT *************** //

	// Eval the Average Time Performing Iterative 1D FFT and output to a file
	//eval_FFT2d_4Data(cifar10,	// Dataset
	//				 fft_2d,     // FFT method to test
	//				 2,			// warm up runs (excluded in eval)
	//				 5,			// testruns to take the average of
	//				 true,		// if write to file
	//				 "our2d");  // base filename



	//*********************************************************//
    //fftw_free(vec_in);
    //fftw_free(vec_out); // no need to reclaim memory for stack allocation



    // ********** Testing 2D DataLoader *********//
    /*
    // Test getting an image
    int imageIndex = 0;
    auto originalImage = dataset.getImage(imageIndex);

    // Modify the image - for example, set all pixels to a specific grayscale value
    complex<double>** modifiedImage = new complex<double>*[32];
    for (int i = 0; i < 32; ++i) {
        modifiedImage[i] = new complex<double>[32];
        for (int j = 0; j < 32; ++j) {
            modifiedImage[i][j] = complex<double>(50, 0); // Set to grayscale value 50
        }
    }
    // Set the modified image in the dataset
    dataset.setImage(imageIndex, modifiedImage, false);
    // Retrieve the modified image to verify changes
    std::complex<double>** retrievedImage = dataset.getImage(imageIndex);
    // Print out some pixel values to verify the change
    std::cout << "Original and Modified Image Pixel Values:" << std::endl;
    for (int i = 0; i < 5; ++i) {
        std::cout << "Pixel " << i << ": " << originalImage[i / 32][i % 32].real() 
        << ", " << retrievedImage[i / 32][i % 32].real() << std::endl;
    }
    // Cleanup dynamically allocated memory
    for (int i = 0; i < 32; ++i) {
        delete[] originalImage[i];
        delete[] modifiedImage[i];
        delete[] retrievedImage[i];
    }
    delete[] originalImage;
    delete[] modifiedImage;
    delete[] retrievedImage;
    */
    //*********************************************//
}
