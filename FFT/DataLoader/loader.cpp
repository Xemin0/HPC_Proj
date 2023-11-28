/*
 * Loading Data from the specified path
 */


#include <fstream> // File operations
#include <stdexcept>
#include <complex>

#include "../lib/loader.h"
//#include <string>

using namespace std;



// Constructor 
Dataset::Dataset(const std::string& dataFilename, const std::string& dimFilename){
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
    fft_data = new complex<double>[rows * cols * depth];

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


Dataset::~Dataset()
{
    delete[] data;
    delete[] cdata;
    delete[] fft_data;
}


// Accesoor for the dimensions
void Dataset::getDimensions(int& r, int& c, int& d) const{
    r = rows; c = cols; d = depth;
}

// Get an element from the dataset (long vector) 
complex<double> Dataset::getElement(unsigned int i, unsigned int j, unsigned int k, bool isComplex) const{
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


