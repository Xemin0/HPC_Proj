#include <iostream>
#include <vector>
#include <complex>
#include <cmath>

const double PI = 3.14159265358979323846;

typedef std::complex<double> Complex;
typedef std::vector<Complex> ComplexVector;

// Cooley-Tukey FFT recursive function
void fft(ComplexVector& x) {
    size_t N = x.size();
    if (N <= 1) {
        return;
    }

    // Divide
    ComplexVector even(N / 2), odd(N / 2);
    for (size_t i = 0; i < N / 2; ++i) {
        even[i] = x[i * 2];
        odd[i] = x[i * 2 + 1];
    }

    // Conquer
    fft(even);
    fft(odd);

    // Combine
    for (size_t k = 0; k < N / 2; ++k) {
        Complex t = std::polar(1.0, -2 * PI * k / N) * odd[k];
        x[k] = even[k] + t;
        x[k + N / 2] = even[k] - t;
    }
}

int main() {
    // Signal parameters
    const size_t N = 256; // Signal size (power of 2)
    const double freq1 = 5.0; // Frequency of the first sine wave
    const double freq2 = 20.0; // Frequency of the second sine wave

    // Generate dummy signal (sum of two sinusoids)
    ComplexVector signal(N);
    for (size_t i = 0; i < N; ++i) {
        double t = static_cast<double>(i) / N;
        signal[i] = Complex(std::sin(2 * PI * freq1 * t) + std::sin(2 * PI * freq2 * t), 0);
    }

    // Perform FFT
    fft(signal);

    // Output the FFT result
    for (size_t i = 0; i < signal.size(); ++i) {
        std::cout << "FFT[" << i << "] = " << signal[i] << std::endl;
    }

    return 0;
}
