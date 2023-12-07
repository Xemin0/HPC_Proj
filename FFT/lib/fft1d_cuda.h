/*
 * fft1d_cuda.h
 * 
 * Description: 1D FFT CUDA Kernels and Corrosponding Kernel Launching and Wrapping Functions
 *
 *      x: N dimensional vector
 *      N: len(x)
 */

#pragma once

typedef std::complex<double> Complex;

__global__ void bitReverse_kernel(Complex *d_x, int N);

// ###########################################

__global__ void fft1d_kernel(Complex *d_x, int N);

void fft1d_device(Complex *d_x, int N);

void fft1d_cu(Complex *h_x, int N);
