## Cooley-Tukey FFT methods

## 1-D FFT
#### List of Methods Available

- `fft_re_1d`: the Recursive method
- `fft_it_1d`: the Iterative method
- `fft2_1d_wrapper`: the Wrapper function for 1D FFT method from FFTW library

**Check Respective Branch for More**

*Recommended format*

`void FFT_method(Complex *vec, int N)`

- `Complex`(dtype) $\equiv$ `complex<double>`
- `N`: length of the array

the FFT result will be stored in the original `vec`


#### 2-D FFT
#### List of Methods Available
- `fft_2d`: 2D method by calling 1D Iterative method
- `fft_2d_wrapper`: the Wrapper function 2D FFT method from FFTW
