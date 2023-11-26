#include <iostream>
#include <array>
#include <complex>
#include <fftw3.h>

const int MAX_SIGNALS = 2;  // Maximum number of signals
const int SIGNAL_LENGTH = 256;  // Length of each signal
const double PI = 3.14159265358979323846;

typedef std::complex<double> Complex;
typedef std::array<Complex, SIGNAL_LENGTH / 2 + 1> ComplexArray;

// FFT computation using FFTW3
void computeFFT(const std::array<double, SIGNAL_LENGTH>& in, ComplexArray& out) {
    fftw_complex *out_complex = fftw_alloc_complex(SIGNAL_LENGTH);
    fftw_plan p = fftw_plan_dft_r2c_1d(SIGNAL_LENGTH, const_cast<double*>(in.data()), out_complex, FFTW_ESTIMATE);

    fftw_execute(p); // execute the plan

    for (size_t i = 0; i < SIGNAL_LENGTH / 2 + 1; ++i) {
        out[i] = std::complex<double>(out_complex[i][0], out_complex[i][1]);
    }

    fftw_destroy_plan(p);
    fftw_free(out_complex);
}

// Spectral coherence calculation
std::array<std::array<double, MAX_SIGNALS>, MAX_SIGNALS> SpectralCoherence(const std::array<std::array<double, SIGNAL_LENGTH>, MAX_SIGNALS>& X) {
    std::array<std::array<double, MAX_SIGNALS>, MAX_SIGNALS> coherence_features{};
    std::array<ComplexArray, MAX_SIGNALS> fft_results{};

    // Compute FFT for each signal
    for (int i = 0; i < MAX_SIGNALS; ++i) {
        computeFFT(X[i], fft_results[i]);
    }

    // Calculate coherence
    for (int i = 0; i < MAX_SIGNALS; ++i) {
        for (int j = i; j < MAX_SIGNALS; ++j) {
            double sum_coherence = 0;
            for (int k = 0; k < SIGNAL_LENGTH / 2 + 1; ++k) {
                auto Pxx = std::norm(fft_results[i][k]);
                auto Pyy = std::norm(fft_results[j][k]);
                auto Pxy = fft_results[i][k] * std::conj(fft_results[j][k]);

                double Cxy = std::norm(Pxy) / (Pxx * Pyy);
                sum_coherence += Cxy;
            }
            double mean_coherence = sum_coherence / (SIGNAL_LENGTH / 2 + 1);
            coherence_features[i][j] = mean_coherence;
            coherence_features[j][i] = mean_coherence; // Symmetric assignment
        }
    }

    return coherence_features;
}

int main() {
    // Example usage with dummy signals
    std::array<std::array<double, SIGNAL_LENGTH>, MAX_SIGNALS> X{};
    
    // Fill with dummy data (simple sine waves for illustration)
    for (int i = 0; i < SIGNAL_LENGTH; ++i) {
        X[0][i] = sin(2 * PI * i / SIGNAL_LENGTH);
        X[1][i] = sin(2 * PI * i / SIGNAL_LENGTH + PI / 4); // Phase-shifted
    }

    auto coherence = SpectralCoherence(X);

    // Output coherence values
    for (int i = 0; i < MAX_SIGNALS; ++i) {
        for (int j = 0; j < MAX_SIGNALS; ++j) {
            std::cout << "Coherence between signal " << i << " and signal " << j << ": " 
                      << coherence[i][j] << std::endl;
        }
    }

    return 0;
}
