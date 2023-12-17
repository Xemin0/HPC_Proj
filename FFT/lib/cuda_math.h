/*
 * Math Funcs on DEVICE
 */

#ifndef LIB_CUDA_MATH_H_
#define LIB_CUDA_MATH_H_

#include <cuda.h>
#include <cuComplex.h>


__device__ cuDoubleComplex pow_cuDoubleComplex(cuDoubleComplex z, int n);


#endif /* LIB_CUDA_MATH_H_ */
