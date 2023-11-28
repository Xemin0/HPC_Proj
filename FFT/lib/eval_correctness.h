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

void FFT1d_4Data(Dataset& ds, FuncPtr func = fft_it_1d,
				 bool toFile = true, std::string filename = "our1d_iter.txt");
// Testing 1D FFT methods on loaded Data

#endif /* LIB_EVAL_CORRECTNESS_H_ */
