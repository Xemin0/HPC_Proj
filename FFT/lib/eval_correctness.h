/*
 * Subroutines to Verify the Correctness of FFT methods
 *
 * ## Default Output Path will be in `./Data/Results/`
 */

#ifndef LIB_EVAL_CORRECTNESS_H_
#define LIB_EVAL_CORRECTNESS_H_

#include "../lib/loader.h"

void FFT1d_4Data(Dataset& ds, bool ifIter = true, bool toFile = true, std::string root_path = "./Data/Results/");
// Testing 1D FFT methods on loaded Data

#endif /* LIB_EVAL_CORRECTNESS_H_ */
