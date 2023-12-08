/*
 * Iterative Method of Cooley-Tukey FFT Algorithm
 * 
 */


#include <complex>
#include <cmath>

#include <iostream>

using namespace std;

typedef std::complex<double> Complex;
const double PI = 3.14159265358973238460;

// ** This Method Overwrites the Input ** //
// ** Remember to create a copy of the input ** //

// bit reverse
void bitReverse(Complex *x, int N) // Effectively Reverse the Bits
{
    for (int i = 1, j = 0; i < N; i++) // ## j is dependent on that from the previous loop step, so this is not parallelizable
    {
        int bit = N >> 1;       // Right-Shifting or Divided by 2
        for (; j&bit; bit >>= 1)// Loop Until j and bit don't have common bits 
        {
            j ^= bit;           // Marking the locations of the bits that are different
        }
        j ^= bit;
        if (i < j) 
            std::swap(x[i], x[j]);
    }
}

// Iterative 1-D FFT
void fft_it_1d(Complex *x, int N)
{
    bitReverse(x, N);
    for (int len = 2; len <= N; len <<= 1) // Butterfly Step (Stages; not parallelizable)
    {
        double angle = -2*PI/len;
        Complex wlen(cos(angle), sin(angle));
        for (int i = 0; i < N; i += len) // Forwarding with a stride: parallelizable
        {
            cout << "for len = " << len << "At segment Idx = " << i << endl;
            Complex w(1);
            for (int j = 0; j < len/2; j++)
            {
                cout << "local tid = " << j << endl;

                cout << "uid = " << i + j << ", vid = " << i + j + len/2 << endl;
                cout << endl;
                Complex u = x[i+j];
                Complex v = x[i + j + len/2]*w;
                x[i+j] = u + v;
                x[i + j + len/2] = u - v;
                w *= wlen;
            }
        }
    }
}
