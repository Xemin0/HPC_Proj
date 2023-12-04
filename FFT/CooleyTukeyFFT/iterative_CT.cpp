/*
 * Iterative Method of Cooley-Tukey FFT Algorithm
 * 
 */


#include <complex>
#include <cmath>

using namespace std;

typedef std::complex<double> Complex;
const double PI = 3.14159265358973238460;

// ** This Method Overwrites the Input ** //
// ** Remember to create a copy of the input ** //

// bit reverse
void bitReverse(Complex *x, int N)
{
    for (int i = 1, j = 0; i < N; i++)
    {
        int bit = N >> 1;
        for (; j&bit; bit >>= 1)
        {
            j ^= bit;
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
    for (int len = 2; len <= N; len <<= 1)
    {
        double angle = -2*PI/len;
        Complex wlen(cos(angle), sin(angle));
        for (int i = 0; i < N; i += len)
        {
            Complex w(1);
            for (int j = 0; j < len/2; j++)
            {
                Complex u = x[i+j];
                Complex v = x[i + j + len/2]*w;
                x[i+j] = u + v;
                x[i + j + len/2] = u - v;
                w *= wlen;
            }
        }
    }
}


