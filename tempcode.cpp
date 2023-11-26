#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <cmath>
#include <stdexcept>

const double PI = 3.14159265358979323846;

typedef std::complex<double> Complex;
typedef std::vector<Complex> ComplexVector;

// Cooley-Tukey FFT recursive function
void fft(ComplexVector& x) {
    size_t N = x.size();
    if (N <= 1) return;

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

class Dataset {
private:
    std::vector<double> data;
    int rows, cols, depth;

public:
    // Constructor that loads the data
    Dataset(const std::string& dataFilename, const std::string& dimFilename) {
        // Read dimensions
        std::ifstream dimFile(dimFilename);
        if (!dimFile.is_open()) {
            throw std::runtime_error("Cannot open dimension file.");
        }
        dimFile >> rows >> cols >> depth;
        dimFile.close();

        // Reserve space for the data
        data.resize(rows * cols * depth);

        // Read binary data
        std::ifstream dataFile(dataFilename, std::ios::binary);
        if (!dataFile.is_open()) {
            throw std::runtime_error("Cannot open data file.");
        }
        dataFile.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(double));
        dataFile.close();
    }

    // Accessor for the dimensions
    void getDimensions(int& r, int& c, int& d) const {
        r = rows; c = cols; d = depth;
    }

    // Function to get an element from the dataset
    double getElement(int i, int j, int k) const {
        if (i < 1 || i > rows || j < 1 || j > cols || k < 1 || k > depth) {
            throw std::out_of_range("Index out of range.");
        }
        // Convert 1-based indices to 0-based indices for internal storage
        int zeroBasedI = i - 1;
        int zeroBasedJ = j - 1;
        int zeroBasedK = k - 1;
        // Calculate index for column-major order
        return data[zeroBasedK * (rows * cols) + zeroBasedJ * rows + zeroBasedI];
    }
};

int main() {
    try {
        // Initialize dataset
        Dataset finger1("finger1_data.bin", "finger1_dimensions.txt");

        int rows, cols, depth;
        finger1.getDimensions(rows, cols, depth);

        // Prepare to write FFT results to file
        std::ofstream fftFile("fft_results.txt");

        // Process FFT for each channel
        for (int i = 1; i <= rows; ++i) {
            ComplexVector signal;
            for (int j = 1; j <= cols; ++j) {
                for (int k = 1; k <= depth; ++k) {
                    signal.push_back(Complex(finger1.getElement(i, j, k), 0));
                }
            }

            // Perform FFT
            fft(signal);

            // Store the FFT result for the current channel
            for (size_t idx = 0; idx < signal.size(); ++idx) {
                fftFile << "Channel " << i << ", FFT[" << idx << "] = " << signal[idx] << std::endl;
            }
        }

        // Close the file
        fftFile.close();

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
