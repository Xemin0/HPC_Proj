/*
 * 1D FFT Optimized with CUDA
 *
 * - Kernels
 * - Kernel Launching Methods
 * - 1D FFT_CUDA Wrapper Function
 */

#include <cuda.h>
#include <cmath>

#include "../lib/iterative_CT.h"

#define BLOCK_SIZE 64 // *** Subject to Change
#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

typedef std::complex<double> Complex;
const double PI = 3.14159265358973238460;

// Kernel for BitReverse
// *** If it's worth to define bitReverse as a kernel ?? *** //
// *** parallelization may not be worthy for small data size *** //
// *** as the overhead of creating threads will outweigh the performance improvement *** //
// *** and Coalesced Memory Accesses may not be guaranteed ** //
__global__ void bitReverse_kernel(Complex *d_x, int N){
    /*
     * (Effectively?) Reverse the Bits of a Vector
     *
     *  - N: length of the vector x; a power of 2
     * 
     * ### Warp Level __shfl_sync may not be efficient ### 
     * Only (N/2 - 1) elements needed to initiate the swaps???
     * since each elememt is only swapped once
     */
    unsigned int startIdx = blockIdx.x * blockDim.x + threadIdx.x + 1; // Skipping the first element
    unsigned int stride = blockDim.x * gridDim.x;

    // Tiling
    // Less likely, but just in case the total number of threads = stride < N
    for (int i = startIdx; i < N - 1; i += stride){
        int j = 0;
        for (int k = 0; k < log2(N); ++k) // for each bit
            if (i & (1 << k))
                j |= (N >> (k+1));

        if (i < j){ // Swap elements at i and j
            Complex tmp = d_x[i];
            d_x[i] = d_x[j];
            d_x[j] = tmp;
        }
    }
}

// The Corresponding Kernel Launching Method


// ###########################################

__global__ void fft1d_kernel(Complex *d_x, int N){
    /*
     * 1 D FFT Kernel (** Currently Require blockDim.x = N/2 **)
     * 
     * stages : iterations for different len 
     * segment: at each stage, the vector is segmented into parts of size len
     *          each segment is handled by len/2 threads
     *          each thread is handling each butterfly pair 
     */
    // ###
    // Bit Reverse Part
    // ###
    unsigned int startIdx = blockIdx.x * blockDim.x + threadIdx.x + 1; // skipping the first element
    unsigned int stride = blockDim.x * gridDim.x;

    // Tiling
    // less likely, but just in case the total number of threads = stride < N
    for (int i = stratIdx; i < N - 1; i += stride){
        int j = 0;
        for (int k = 0; k < log2(N); ++k) // for each bit
            if (i & (1 << k))
                j |= (N >> (k+1));

        if (i < j){ // Swap element at i and j
            Complex tmp = d_x[i];
            d_x[i] = d_x[j];
            d_x[j] = tmp;
        }
    }
    // ###
    // Butterfly Step (Reworked from the Original Iterative Method)
    // ###

    /*
     * Assign len/2 threads to handle len/2 pairs of butterfly operations for each segment
     * Memory Access Pattern: Because the whole vector is accessed for each stage
     * 1. Load vector into a shared memory (**Require blockDim.x >= N** otherwise exchanging data between blocks would further introduce overheads) Shared memory limitation: 48kb or 96kb while Complex is of 16bytes
     * 2. Warp-Level Primitive `__shfl_sync()` because at each stage len is different, managing warps would be difficult
     * 3. A combination of the above two
     */
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // ### blockDim.x = N/2 with one Block

    __shared__ Complex x_shared0[N/2]; // shared memory for d_x First Half
    __shared__ Complex x_shared1[N/2]; // shared memory for d_x Second Half

    // Load d_x to be reused for each stage
    if (idx < N / 2){
        x_shared0[idx] = d_x[idx]; 
        x_shared1[idx] = d_x[idx + N/2];
    }

    __syncthreads();

    // Each Stage
    for (int len = 2; len <= N; len <<= 1){
        double angle = -2 * PI / len;
        Complex wlen(cos(angle), sin(angle));

        if (idx < N/2){ // boundary check for each thread ** half of threads are idling
            int segment_idx = idx / (len/2); // len/2 threads for each segment
            int local_tid = idx % (len/2);
            int segment_start = segment_idx * len;

            Complex w = pow(wlen, local_tid);

            int u_idx = segment_start + local_tid;
            int v_idx = u_idx + len/2;

            Complex u = (u_idx < N/2) ? x_shared0[u_idx] : x_shared1[u_idx - N/2];
            Complex v = (v_idx < N/2) ? x_shared0[v_idx] : x_shared1[v_idx - N/2];

            if (u_idx < N/2)
                x_shared0[u_idx] = u + v;
            else
                x_shared1[u_idx - N/2] = u + v;

            if (v_idx < N/2)
                x_shared0[v_idx] = u - v;
            else
                x_shared1[v_idx] = u - v;
        }
    }

    __syncthreads();

    // Write back to Global Memory
    if (idx < N / 2){
        d_x[idx] = x_shared0[idx];
        d_x[idx + N/2] = x_shared1[idx];
    }


void fft1d_device(Complex *d_x, int N)
{
    /*
     * Kernel Launching Method of fft_1d_kernel
     * both the input and the output stay on DEVICE
     * 
     * As the Butterfly Step requires N/2 threads
     * we will fix BlockDim = N/2 and GridDim = 1
     */
    dim3 nthreads(N/2, 1, 1);   // BlockDim
    dim3 nblocks(1, 1, 1);      // GridDim

    fft1d_kernel <<< nblocks, nthreads, 0, 0 >>> (d_x, N);
}


void fft1d_cu(Complex *h_x, int N)
{
    /*
     * Wrapper Function of 1D FFT with CUDA
     *
     * Assumes Data Allocated on HOST 
     * 1. Transfer Data to DEVICE
     * 2. Launch the Kernel
     * 3. Copy Back the Results
     */
    // Allocate memory on DEVICE
    Complex *d_x;
    cudaMalloc( (void**) &d_x, sizeof(Complex)*N );

    // Copy the vector from HOST to DEVICE
    cudaMemcpy(d_x, h_x, sizeof(Complex) * N, cudaMemcpyHostToDevice);

    // Launch the Kernel
    fft1d_device(d_x, N);

    // Copy back the result from DEVICE to HOST
    cudaMemcpy(h_x, d_x, sizeof(Complex) * N, cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_x);
}

// ###########################################



