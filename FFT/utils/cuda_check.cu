/*
 * Cuda Debug Utilities
 */

#include <cuda.h>
#include <string>
#include <stdio.h>

void last_cuda_error(std::string event)
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
        fprintf(stderr, "CUDA Error at %s: %s\n", event.c_str(), cudaGetErrorString(err));
}
