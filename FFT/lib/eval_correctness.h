/*
 * Subroutines to Verify the Correctness of FFT methods
 *
 * ## Default Output Path will be in `./Data/Results/`
 */

#ifndef LIB_EVAL_CORRECTNESS_H_
#define LIB_EVAL_CORRECTNESS_H_

#include "../lib/iterative_CT.h"
#include "../lib/loader.h"

typedef void (*FuncPtr)(Complex*, int);
typedef void (*FuncPtr2)(Complex**, int, int);

void FFT1d_4Data(Dataset1D& ds, FuncPtr func = fft_it_1d,
				 bool toFile = true, std::string filename = "our1d_iter.txt");

void FFT2d_4Data(Dataset2D& ds, FuncPtr2 func = fftw_2d_wrapper, 
				 bool toFile = true, std::string filename = "out2d_fftw.txt");
// Testing 1D FFT methods on loaded Data

#endif /* LIB_EVAL_CORRECTNESS_H_ */
