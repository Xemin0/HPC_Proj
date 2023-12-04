#include <complex>
#include <cmath>
#include <omp.h>  // Include OpenMP header

using namespace std;

typedef std::complex<double> Complex;
const double PI = 3.14159265358973238460;

// // bit reverse --> Already defined in iterative_CT.cpp
void bitReverse2(Complex *x, int N)
{
    for (int i = 1, j = 0; i < N; i++)
    {
        int bit = N >> 1;
        for (; j & bit; bit >>= 1)
        {
            j ^= bit;
        }
        j ^= bit;
        if (i < j)
            std::swap(x[i], x[j]);
    }
}

// Iterative 1-D FFT with OpenMP
void fft_it_1d_openMP(Complex *x, int N)
{
    bitReverse2(x, N);

    // Parallelizing the outer loop
    for (int len = 2; len <= N; len <<= 1)
    {
        double angle = -2 * PI / len;
        Complex wlen(cos(angle), sin(angle));

        // Use OpenMP to parallelize this loop
        #pragma omp parallel for default(none) shared(x, N, len, wlen)
        for (int i = 0; i < N; i += len)
        {
            Complex w(1);
            for (int j = 0; j < len / 2; j++)
            {
                Complex u = x[i + j];
                Complex v = x[i + j + len / 2] * w;
                x[i + j] = u + v;
                x[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
}
