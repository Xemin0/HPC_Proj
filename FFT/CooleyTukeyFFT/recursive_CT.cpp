/*
 * Recursive Method of Cooley-Tukey FFT Algorithm
 * 
 * 1. Independent Recursive Step: Divide the sequence into Odd and Even elements
 * 2. Combining Step: Dot Product. 
 */


#include <complex>

const double PI = 3.14159265358979323846;

typedef std::complex<double> Complex;

// Recursive 1-D FFT
void fft_re_1d(Complex *x, int N)
{
	if (N <= 1) return;

	// Divide
	// Automatic memory management by std::unique_ptr
	Complex even[N/2], odd[N/2];

	for (int i = 0; i < N/2; i++)
	{
		even[i] = x[i * 2];
		odd[i] = x[i * 2 + 1];
	}

	// recursive step
	fft_re_1d(even, N/2);
	fft_re_1d(odd, N/2);

	// Combine Step
	for (int i = 0; i < N/2; i++)
	{
		Complex t = std::polar(1.0, -2 * PI * i / N) * odd[i];
		x[i] = even[i] + t;
		x[i + N/2] = even[i] - t;
	}
}
