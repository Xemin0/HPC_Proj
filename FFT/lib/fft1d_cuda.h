/*
 * fft1d_cuda.h
 * 
 * Description: 1D FFT CUDA Kernels and Corrosponding Kernel Launching and Wrapping Functions
 *
 *      x: N dimensional vector
 *      N: len(x)
 */

#pragma once

#include <cuComplex.h>
#include <string>

typedef std::complex<double> Complex;

__global__ void bitReverse_kernel(cuDoubleComplex *d_x, int N);

void bitReverse_device(cuDoubleComplex *d_x, int N);

void bitReverse_cu(Complex *h_x, int N);
// ###########################################

__global__ void fft1d_kernel(cuDoubleComplex *d_x, int N);

void fft1d_device(cuDoubleComplex *d_x, int N);

void fft1d_cu(Complex *h_x, int N);

// ###########################################
// #### Kernels and Kernel Launching Methods for Batch Input
// ####
// #### - fft1d_batch_cu    : Sequentially call the kernels for each vector in batch 
// #### - fft1d_batch_cu2   : With kernels that handles the batch with thread blocks (Tiling)
// #### - fft1d_batch_cu3   : With Streams
// ###########################################

void fft1d_batch_cu(Complex *h_x, int N, int batch_size);

// ###########################################
__global__ void fft1d_batch_kernel(cuDoubleComplex *d_x, int N, int batch_size, 
                                   int n_blocks);

void fft1d_batch_device(cuDoubleComplex *d_x, int N, int batch_size,
                        int n_blocks);

void fft1d_batch_cu2(Complex *h_x, int N, int batch_size,
                     int n_blocks);

// ###########################################
