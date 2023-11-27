/*
 * Loading Data from the specified path
 */

#ifndef LOADER_H
#define LOADER_H

#include <complex>
//#include <string>


class Dataset {
private:
    double *data; // column major storage
    int rows, cols, depth;
    std::complex<double> *cdata;  // Column major storage

public:
    std::complex<double> *fft_data;  // Column major storage
    // Constructor 
    Dataset(const std::string& dataFilename = "../Data/finger1_data.bin", const std::string& dimFilename = "finger1_dimensions.txt");
    // Load data along with dimensions from files in ../Data

    // Accesoor for the dimensions
    void getDimensions(int& r, int& c, int& d) const;

    // Get an element from the dataset (long vector) 
    std::complex<double> getElement(unsigned int i, unsigned int j, unsigned int k) const;
};
#endif // LOADER_H
