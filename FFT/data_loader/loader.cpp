/*
 * Loading Data from the specified path
 *
 **** Matrix Data output from Matlab is stored in a Column-major fassion ****
 */

#include <iostream>
#include <fstream> // File operations
#include <stdexcept>
#include <complex>

#include "../lib/loader.h"
//#include <string>

using namespace std;



// Constructor 
Dataset1D::Dataset1D(const std::string& dataFilename, const std::string& dimFilename){
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


Dataset1D::~Dataset1D()
{
    delete[] data;
    delete[] cdata;
    delete[] fft_data;
}


// Accesoor for the dimensions
void Dataset1D::getDimensions(int& r, int& c, int& d) const{
    r = rows; c = cols; d = depth;
}

// Get an element from the dataset (long vector) 
// indices i, j, k starting from 1 to agree with that from Matlab
complex<double> Dataset1D::getElement(unsigned int i, unsigned int j, unsigned int k, 
                                    bool isFFT) const{
    if (i < 1 || i > rows || j < 1 || j > cols || k < 1 || k > depth)
        throw out_of_range("Index out of range.");

    // Convert 1-based indices to 0-based indices for internal storage
    // matlab array to cpp array indices
    // **** NECESSARY?? ****
    int zeroBasedI = i - 1;
    int zeroBasedJ = j - 1;
    int zeroBasedK = k - 1;

    if (isFFT)
        return fft_data[zeroBasedK * (rows * cols) + zeroBasedJ * rows + zeroBasedI];
    else
        return cdata[zeroBasedK * (rows * cols) + zeroBasedJ * rows + zeroBasedI];
}

// Set an element
// indices i, j, k starting from 1 to agree with that from Matlab
void Dataset1D::setElement(complex<double> val, unsigned int i, unsigned int j, unsigned int k, bool isFFT)
{
    if (i < 1 || i > rows || j < 1 || j > cols || k < 1 || k > depth)
        throw out_of_range("Index out of range.");

    if (isFFT)
        fft_data[(k-1) * (rows * cols) + (j-1) * rows + (i-1)] = val;
    else
        cdata[(k-1) * (rows * cols) + (j-1) * rows + (i-1)] = val;
} 

// ###### Implement Dataset2D class ######

// Constructor 
Dataset2D::Dataset2D(const std::string& dataFilename, const std::string& dimFilename){
    // Load data along with dimensions from files in ../Data

    // Read dimensions 
    ifstream dimFile(dimFilename);
    if (!dimFile.is_open())
        throw runtime_error("Cannot open dimension file.");

    dimFile >> rows >> cols >> channels >> numImages; // TODO: channels might not be necessary!!
    
    dimFile.close();
    imageSize = rows * cols;

    images = new complex<double>[numImages * imageSize];
    fft_data = new complex<double>[numImages * imageSize];
    // for (int i = 0; i < channels; ++i) {
    //         images[i] = new std::complex<double>[imageSize * numImages];
    //     }

    // Read binary data
    std::ifstream dataFile(dataFilename, std::ios::binary);
        if (!dataFile.is_open()) {
            throw std::runtime_error("Cannot open data file.");
        }

    unsigned char buffer[imageSize * 3]; // Buffer to store RGB values for one image
    for (int img = 0; img < numImages; ++img) {
        // Skip the label byte at the beginning of each image
        dataFile.ignore(1);

        // Read the entire RGB data for one image
        dataFile.read(reinterpret_cast<char*>(buffer), sizeof(buffer));
        
        // Convert each pixel to grayscale and store it
        for (int i = 0; i < imageSize; ++i) {
            unsigned char r = buffer[i];
            unsigned char g = buffer[i + imageSize];
            unsigned char b = buffer[i + 2 * imageSize];

            // Luminosity method: 0.21 R + 0.72 G + 0.07 B
            double gray = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            images[img * imageSize + i] = std::complex<double>(gray, 0.0);
        }
    }

    dataFile.close();
}

// Destructor
Dataset2D::~Dataset2D() {
    delete[] images;
    delete[] fft_data;
}

// Get a specific element (pixel value) from the dataset
complex<double> Dataset2D::getElement(unsigned int imageIndex, unsigned int row, unsigned int col, bool isFFT) const {
    if (imageIndex < 0 || imageIndex >= numImages || row < 0 || row >= rows || col < 0 || col >= cols)
        throw out_of_range("Index out of range.");

    if (isFFT)
        return fft_data[imageIndex * rows * cols + row * cols + col];
    else
        return images[imageIndex * rows * cols + row * cols + col];
}

// Set a specific element (pixel value) in the dataset
void Dataset2D::setElement(complex<double> val, unsigned int imageIndex, unsigned int row, unsigned int col, bool isFFT) {
    if (imageIndex < 0 || imageIndex >= numImages || row < 0 || row >= rows || col < 0 || col >= cols)
        throw std::out_of_range("Index out of range.");

    if (isFFT)
        fft_data[imageIndex * rows * cols + row * cols + col] = val;
    else
        images[imageIndex * rows * cols + row * cols + col] = val;
}

// Accessor for the dimensions
void Dataset2D::getDimensions(int& r, int& c, int& nImages) const{
    r = rows; c = cols; nImages = numImages;
}

// Get the 2d array for a specific image
complex<double>** Dataset2D::getImage(unsigned int imageIndex, bool isFFT) const {
    if (imageIndex < 0 || imageIndex >= numImages)
        throw std::out_of_range("Image index out of range.");

    std::complex<double>** image = new std::complex<double>*[rows];
    if (isFFT) {
        for (int i = 0; i < rows; ++i) {
            image[i] = new std::complex<double>[cols];
            for (int j = 0; j < cols; ++j) {
                image[i][j] = fft_data[imageIndex * rows * cols + i * cols + j];
            }
        }
    } 
    else {
        for (int i = 0; i < rows; ++i) {
            image[i] = new std::complex<double>[cols];
            for (int j = 0; j < cols; ++j) {
                image[i][j] = images[imageIndex * rows * cols + i * cols + j];
            }
        }
    }
    return image;
}
   
// Set the pixel values for a specific image
void Dataset2D::setImage(unsigned int imageIndex, std::complex<double>** image, bool isFFT) {
    if (imageIndex < 0 || imageIndex >= numImages)
        throw std::out_of_range("Image index out of range.");

    if (isFFT) {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                fft_data[imageIndex * rows * cols + i * cols + j] = image[i][j];
            }
        }
    }
    else {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                images[imageIndex * rows * cols + i * cols + j] = image[i][j];
            }
        }
    }
}
