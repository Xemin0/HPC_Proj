## Compile with `FFTW` library
*Change the library path as needed*

Navigate to `./FFT/` Folder then run following bash command (using `g++` should suffice) 

```bash
g++-13 ./DataLoader/loader.cpp main.cpp -o It_CT.o -I/usr/local/Cellar/fftw/3.3.10_1/include -L/usr/local/Cellar/fftw/3.3.10_1/lib -lfftw3
```

## Run 
`./It_CT.o`
