/*
 * Subroutines to Verify the Correctness of FFT methods
 *
 * ## Default Output Path will be in `./Data/Results/`
 */

#ifndef LIB_EVAL_CORRECTNESS_H_
#define LIB_EVAL_CORRECTNESS_H_

#include "../lib/iterative_CT.h"
#include "../lib/fft1d_cuda.h" // fft1d_batch_cu()
                               // fft1d_batch_cu2()
#include "../lib/loader.h"

typedef void (*FuncPtr)(Complex*, int);
                                    // 1D Array, Length 
typedef void (*FuncPtrBatch)(Complex*, int, int, int, int); // for 1D Batch FFT methods      
                                    // 1D Array, Length, BatchSize, num_blocks, num_streams
typedef void (*FuncPtr2)(Complex**, int, int);

void FFT1d_4Data(Dataset1D& ds, FuncPtr func = fft_it_1d,
				 bool toFile = true, std::string filename = "our1d_iter.txt");

void FFT1d_4BatchData(Dataset1D& ds, FuncPtrBatch func = fft1d_batch_cu2, 
                      int n_blocks = 1,
                      int n_streams = 0,
                      bool toFile = true, std::string filename = "our1d_batch.txt");


void FFT2d_4Data(Dataset2D& ds, FuncPtr2 func = fftw_2d_wrapper, 
				 bool toFile = true, std::string filename = "our2d_fftw.txt");
// Testing 1D FFT methods on loaded Data

#endif /* LIB_EVAL_CORRECTNESS_H_ */
