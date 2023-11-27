/*
 * Loading Data from the specified path
 */


#include <fstream> // File operations
#include <stdexcept>
#include <complex>
//#include <string>

using namespace std;


class Dataset {
private:
    double *data; // 1D array
    int rows, cols, depth;
    complex<double> *cdata; // 1D array 0 imag part by default

public:
    complex<double> *fft_data; // 1D array to store fft result
    // Constructor 
    Dataset(const std::string& dataFilename = "../Data/finger1_data.bin", const std::string& dimFilename = "finger1_dimensions.txt"){
        // Load data along with dimensions from files in ../Data
    
        // Read dimensions 
        ifstream dimFile(dimFilename);
        if (!dimFile.is_open())
            throw runtime_error("Cannot open dimension file.");

        dimFile >> rows >> cols >> depth;
        dimFile.close();

        // Allocate Memory for 
        data = new double[rows * cols * depth];
        cdata = new complex<double>[rows * cols * depth];

        // Read binary data
        ifstream dataFile(dataFilename, std::ios::binary);
        if (!dataFile.is_open())
            throw runtime_error("Cannot open data file.");
        // expected char from binary file
        dataFile.read(reinterpret_cast<char*>(data), rows * cols * depth * sizeof(double));
        dataFile.close();

        // convert data to complex data for FFT
        for (int i = 1; i <= rows * cols * depth; i++)
            cdata[i] = complex<double>(data[i], 0);
    }

    // Accesoor for the dimensions
    void getDimensions(int& r, int& c, int& d) const{
        r = rows; c = cols; d = depth;
    }

    // Get an element from the dataset (long vector) 
    complex<double> getElement(unsigned int i, unsigned int j, unsigned int k, bool isComplex = true) const{
        if (i < 1 || i > rows || j < 1 || j > cols || k < 1 || k > depth)
            throw out_of_range("Index out of range.");

        // Convert 1-based indices to 0-based indices for internal storage
        // matlab array to cpp array indices
        // **** NECESSARY?? ****
        int zeroBasedI = i - 1;
        int zeroBasedJ = j - 1;
        int zeroBasedK = k - 1;

        if (isComplex)
            return cdata[zeroBasedK * (rows * cols) + zeroBasedJ * rows + zeroBasedI];
        else
            return data[zeroBasedK * (rows * cols) + zeroBasedJ * rows + zeroBasedI];
            
    }
};

