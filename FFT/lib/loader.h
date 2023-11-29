/*
 * Loading Data from the specified path
 *
 **** Matrix Data output from Matlab is stored in a Column-major fassion ****
 */

#ifndef LIB_LOADER_H_
#define LIB_LOADER_H_

#include <complex>
//#include <string>


class Dataset1D {
private:
    double *data; // column major storage from Matlab
    int rows, cols, depth;
    std::complex<double> *cdata;  // Column major storage; Complex version of Data
    std::complex<double> *fft_data;  // Column major storage; FFTed Data

public:
    // Constructor 
    Dataset1D(const std::string& dataFilename = "./Data/finger1_data.bin", const std::string& dimFilename = "./Data/finger1_dimensions.txt");
    // Load data along with dimensions from files in ../Data

    ~Dataset1D();

    // Accesoor for the dimensions
    void getDimensions(int& r, int& c, int& d) const;

    // Get an element from the dataset (long vector) 
    // indices i, j, k starting from 1 to agree with that from Matlab
    std::complex<double> getElement(unsigned int i, unsigned int j, unsigned int k, bool isFFT = false) const;

    // Set an element
    // indices i, j, k starting from 1 to agree with that from Matlab
    void setElement(std::complex<double> val, unsigned int i, unsigned int j, unsigned int k, bool isFFT = true);
};
#endif // LIB_LOADER_H_
