#!/bin/bash

# Compile fft.cpp
g++ -arch arm64 -o fft It_CooleyTukey.cpp -I/opt/homebrew/Cellar/fftw/3.3.10_1/include -L/opt/homebrew/Cellar/fftw/3.3.10_1/lib -lfftw3

# Check if compilation was successful
if [ $? -eq 0 ]; then
    # Run the program and store output in output.txt
    ./fft $1 $2 > $3
else
    echo "Compilation failed."
fi