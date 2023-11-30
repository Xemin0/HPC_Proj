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


class Dataset2D {
private:
    //double* data; // Column major storage for 2D data
    int rows, cols, channels, numImages, imageSize; // Dimensions of the 2D dataset (the number of rows, cols, and channels for each image and the num of images)

    std::complex<double>* images;
    //std::complex<double>* cdata; // Column major storage for complex data
    std::complex<double>* fft_data; // Column major storage for FFT-transformed data

public:
    // Constructor
    Dataset2D(const std::string& dataFilename = "./Data/cifar_data.bin", const std::string& dimFilename = "./Data/cifar_dimensions.txt");

    ~Dataset2D();

    // Accessor for the dimensions
    void getDimensions(int& r, int& c, int& nImages) const;

    // Get an element from the dataset
    std::complex<double> getElement(unsigned int imgID, unsigned int i, unsigned int j, bool isFFT = false) const;

    // Set an element in the dataset
    void setElement(std::complex<double> val, unsigned int imgID, unsigned int i, unsigned int j, bool isFFT = true);

    // Get the data for a specific image
    std::complex<double>** getImage(unsigned int imageIndex, bool isFFT = false) const;

    // Set the data for a specific image
    void setImage(unsigned int imageIndex, std::complex<double>** image, bool isFFT = true);
};


#endif // LIB_LOADER_H_
