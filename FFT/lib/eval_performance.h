/*
 * Subroutines to Time and Calculate the Performance of FFT methods
 *
 *
 * - time evaluation
 * - FLOPrate evaluation
 *
 * ## Default Output Path will be in `./Data/Results/`, as a `.dat` file
 */

#ifndef LIB_EVAL_PERFORMANCE_H_
#define LIB_EVAL_PERFORMANCE_H_

#include "../lib/iterative_CT.h"
#include "../lib/loader.h"

typedef void (*FuncPtr)(Complex*, int);     // for 1D FFT methods
typedef void (*FuncPtrBatch)(Complex*, int, int, int); // for 1D Batch FFT methods
typedef void (*FuncPtr2)(Complex**, int, int);   // for 2D FFT methods

float time_FFT1d_4Data(Dataset1D& ds, FuncPtr func = fft_it_1d, bool isCPU = true);
// ## May need to return HighPrecisionTimer object
// Time a single run of provided 1D FFT method over the whole dataset in microsecond(us)

float eval_FFT1d_4Data(Dataset1D& ds, FuncPtr func = fft_it_1d,
                      bool isCPU = true,
					  int warmup = 2, int testruns = 5,
				 	  bool toFile = true, std::string filename = "our1d_iter");
// Average Time of 1D FFT methods on the whole Data in microsecond (us)
// and (maybe) calculate FLOPrate
//  "warmup" warm-up runs (excluded in eval)
// and then take the average of "testruns" runs

// ********** 1D FFT For Batch Input Performance Evaluation *********** //

float time_FFT1d_4BatchData(Dataset1D& ds, FuncPtrBatch func = fft1d_batch_cu2,
                            int n_blocks = 4,
                            bool isCPU = true);

float eval_FFT1d_4BatchData(Dataset1D& ds,
                      FuncPtrBatch func = fft1d_batch_cu2,
                      int n_blocks = 4,
                      bool isCPU = true,
                      int warmup = 2, int testruns = 5,
                      bool toFile = true, std::string filename = "our1d_cu");

// ********** 2D FFT Performance Evaluation *********** //

float time_FFT2d_4Data(Dataset2D& ds, FuncPtr2 func, bool isCPU = true);
// ## May need to return HighPrecisionTimer object
//Time a single run of provided 2D FFT method over the whole dataset


float eval_FFT2d_4Data(Dataset2D& ds, FuncPtr2 func, 
                      bool isCPU = true,
                      int warmup = 2, int testruns = 5,
                      bool toFile = true, std::string filename = "our2d");
// Average Performance of 2D FFT for a given Dataset in microsecond (us)

#endif /* LIB_EVAL_PERFORMANCE_H_ */
