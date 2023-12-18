## 1-D FFT 
#### List of Methods Available
- `fft_it_1d`         : 1D FFT Iterative Method
- `fft_re_1d`         : 1D FFT Recursive Method
- `fftw_1d_wrapper`   : 1D FFT method from FFTW
- `fft1d_cu`          : 1D FFT method w. CUDA

- `fft1d_batch_cu`    : 1D FFT for batch input - Sequentially launched kernels
- `fft1d_batch_cu2`   : 1D FFT for batch input - Kernel with batch input   
- ~`fft1d_batch_cu3`~ : ~1D FFT for batch input - Streams~ (Not yet implemented)


*Recommended format*

`void FFT_method(Complex *vec, int N)` 

- `Complex`(dtype) $\equiv$ `complex<double>`
- `N`: length of the array

the FFT result will be stored in the original `vec`


#### 2-D FFT 
#### List of Methods Available
- `fft2d`             : 2D FFT
- `fftw_2d_wrapper`   : 2D FFT method from FFTW
