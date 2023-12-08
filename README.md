# Parallelizing 2D Fast-Fourier Transform Implemented with Cooley-Tukey Algorithm Using CUDA

## File Structure
**Each folder has (and will have) its respective `readme` describing what type of modules should be placed inside**

**Make sure you have the following folders**
- Data Files should be placed in `./FFT/Data/` 
- Output results by methods from `/FFT/eval/` will be output to `/FFT/Data/Results`

## Prerequisite
- [FFTW C/C++ library](https://www.fftw.org/) for validating the results (not necessary)
    - `MacOS` installation: `brew install fftw`
- Most likely a 'Turing Arch' GPU with `CUDA10.2.0` 

## Compile with `FFTW` library
*Change the `FFTW` library path as needed*

### Compilation with Makefile (recommended)
In the `./FFT/` folder use the bash command `make`

*If a fresh build is desired, simply run `make clean` before running `make` or `make all`*

## Run 
`./It_CT.out`

or `nsys profile ./It_CT.out` to profile with NVIDIA NSIGHT

## TO-DOS
### 1D FFT
- ~Design Kernels For 1D FFT~
- ~Implement Kernel Launching Method for 1D FFT~
- ~Validate Correctness~
- Implement `eval_correctness` and `eval_performance` methods for CUDA

### 2D FFT
- ?? Design Kernels for 2D FFT (Nested Calls?)
- ?? Directly Optimize 2D FFT after `bitReverse` step
- Kernel Launching method
- Correctness and Performance

### Misc
- `cudaCheck()`

### Primary Files to Work On
- `CooleyTukeyFFT/*.cu`
- `eval/eval_correctness.cu`
- `eval/eval_performance.cu`
- Corresponding Headers
- Makefile
- `SLURM` script
