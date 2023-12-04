# Parallelizing 2D Fast-Fourier Transform Implemented with Cooley-Tukey Algorithm

## File Structure
**Each folder has (and will have) its respective `readme` describing what type of modules should be placed inside**

**Make sure you have the following folders**
- Data Files should be placed in `./FFT/Data/` 
- Output results by methods from `/FFT/eval/` will be output to `/FFT/Data/Results`


## Prerequisite
- [FFTW C/C++ library](https://www.fftw.org/) for validating the results (not necessary)
    - `MacOS` installation: `brew install fftw`

## Compile with `FFTW` library
*Change the `FFTW` library path as needed*

## Manual Compilation
Navigate to `./FFT/` Folder then run the following bash command

~`g++-13 ./data_loader/loader.cpp main.cpp -o It_CT.o -I/usr/local/Cellar/fftw/3.3.10_1/include -L/usr/local/Cellar/fftw/3.3.10_1/lib -lfftw3`~

### Compilation with Makefile (recommended)
In the `./FFT/` folder use the bash command `make`

*If a fresh build is desired, simply run `make clean` before running `make` or `make all`*

## Run 
`./It_CT.out`

## TO-DOS
### 1D FFT
- ~Generalize `./eval/eval_correctness.cpp` subroutines to take function pointers as its parameter~
- ~Implement subroutines to evaluate performances (time evalution of execution)~

### 2D FFT
- ~2D FFT~
- ~FFTW_2D wrapper~
- ~`eval_performance` for 2D methods~
- ~`eval_correctness` for 2D methods~
- ~Dataloader for 2D data (CIFAR10 grayscale)~

### `OpenMP` v.s. `MPI` v.s. `CUDA`
- Keep Codes in Separate Branches
- `makefile` for each
- `SLURM` script for each
