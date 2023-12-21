/*
 * Subroutines to Verify the Correctness of FFT methods
 *
 * ## Default Output Path will be in `./Data/Results/`
 */


#include <cuda.h>
#include <iostream>
#include <complex>
#include <cmath>
#include <fstream>
#include <stdlib.h>

//#include "../lib/iterative_CT.h"
//#include "../lib/fftw_wrapper.h"
//#include "../lib/fft1d_cuda.h" // fft1d_batch_cu()
                               // fft1d_batch_cu2()

#include "../lib/loader.h"
using namespace std;

typedef complex<double> Complex;

// Define the function pointer
typedef void (*FuncPtr)(Complex*, int);
                                    // 1D Array, Length 
typedef void (*FuncPtrBatch)(Complex*, int, int, int, int); // for 1D Batch FFT methods
                                    // 1D Array, Length, BatchSize, num_blocks, num_streams

// Define the function pointer for 2D FFT functionalities
typedef void (*FuncPtr2)(Complex**, int, int);

void FFT1d_4Data(Dataset1D& ds, FuncPtr func, 
				 bool toFile, std::string filename)
{
 	/*
     * Correctness of FFT for a given Dataset 
     * 
     * - FuncPtr: Function pointer of 1D FFT method
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

void FFT1d_4BatchData(Dataset1D& ds, FuncPtrBatch func, 
                      int n_blocks,
                      int n_streams,
                      bool toFile, std::string filename)
{
    /*
     * Correctness of 1D FFT for a Given Dataset as a Batch Input
     *
     *** Specifically for CUDA Methods
     */
    std::string root_path = "./Data/Results/";

    int rows, cols, depth;   
    ds.getDimensions(rows, cols, depth);
    //cout << "rows = " << rows << endl;
    //cout << "cols = " << cols << endl;
    //cout << "depth = " << depth << endl;

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

    // Allocate for a long vector that stores the truncated vectors
    Complex *all_vecs;
    cudaHostAlloc((void**)&all_vecs, rows*depth * truncated_cols * sizeof(Complex), cudaHostAllocDefault);

    // Copy data into this vector
    for (int i = 0; i < rows; i++)
        for (int k = 0; k < depth; k++)
            for (int j = 0; j < truncated_cols; j++)
                all_vecs[i*k*truncated_cols + j] = ds.getElement(i+1, j+1, k+1);

    // 1D FFT with provided method (as a function pointer)
    func(all_vecs, truncated_cols, rows*depth, n_blocks, n_streams);
    
    // Write the results back to Dataset or the file
    for (int i = 0; i < 1; i++) // **** change 1 to rows
        for (int k = 0; k < 1; k++) // **** change 1 to depth
            for (int j = 0; j < truncated_cols; j++)
                fftFile << "Channel " << i << ", FFT[" << j <<"] = " << all_vecs[i*k*truncated_cols + j] << endl;

    // Clean Up
    fftFile.close();
    cudaFreeHost(all_vecs);
}


//************************************************************//

void FFT2d_4Data(Dataset2D& ds, FuncPtr2 func, 
				 bool toFile, std::string filename)
{
 	/*
     * Correctness of FFT for a given 2D Dataset 
     * 
     * - FuncPtr: Function pointer of 2D FFT method
	 * 			  in the form of `void func(Complex **x, int rows, int cols)` 
     */

	std::string root_path = "./Data/Results/";

    int rows, cols, numImages;   
    ds.getDimensions(rows, cols, numImages);
    cout << "rows = " << rows << endl;
    cout << "cols = " << cols << endl;
    cout << "numImages = " << numImages << endl;

    // Iterative FFT requires input size to be a power of 2
    int truncated_rows = log2(rows);
    truncated_rows = pow(2, truncated_rows);
    int truncated_cols = log2(cols);
    truncated_cols = pow(2, truncated_cols);
    cout << "resized row and column size: \t" << truncated_rows << ", " << truncated_cols << endl;

    // File name preparation
    filename = root_path + filename;

    
    ofstream fftFile(filename); 
    if (!fftFile.is_open()){
        cerr << "Failed to open file for writing!" << endl;
    }
    cout << "Writing results to: " << filename << endl;

    // get data for each image
    Complex **tmpImage = new Complex *[rows];

    // FFT for each image
    for (int imgIdx = 0; imgIdx < 10; imgIdx++) { // ***  change 10 to numImages
    
        tmpImage = ds.getImage(imgIdx); 
        
        // FFT with provided method (as a function pointer)
        func(tmpImage, truncated_rows, truncated_cols);

        // Write Our result to the output fft image 
        // Row Major
        for (int i = 0; i < truncated_rows; i++)
            ds.setImage(imgIdx, tmpImage, true); // true: set image for ds.fft_data
            
        // store the result for current channel
        if (toFile) {      
            for (int i = 0; i < truncated_rows; ++i) {
                for (int j = 0; j < truncated_cols; ++j) {
                    fftFile << "Image " << imgIdx << ", FFT[" << i << ", " << j << "] = " << tmpImage[i][j] << endl;
                }
                fftFile << std::endl; // New line at the end of each row
            }
        }
        for (int i = 0; i < rows; ++i) 
            delete[] tmpImage[i];

        delete[] tmpImage;
    }
    
    fftFile.close();
}


