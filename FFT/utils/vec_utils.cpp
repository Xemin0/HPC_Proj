/*
 * Miscellaneous Utilities for Vectors and Vector Manipulations 
 */



#include <complex>
#include <stdlib.h>
#include <time.h> // seed for random generator
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace std;

typedef std::complex<double> Complex;

// ** The Use of Template Requires Explicit Instantiation after Function Definitions ** //
// Check the end section of this file //

template <typename T>
void show_vec(T *vec, unsigned int N)
{
    for (unsigned int i = 0; i < N; i++)
        cout << *(vec + i) << " ";

    cout << endl;
}

template <typename T>
void copy_vec(T *vec1, T *vec2, unsigned int N)
{
    // Copy vec1 into vec2
    for (int i = 0; i < N; i++)
        vec2[i] = vec1[i];
}


Complex* rand_vec(unsigned int N, double lower_bound = -100, double upper_bound = 100)
{
    /*
     * Generate a random N dimensional complex vector
     *
     * ## Currently:
     *              - Only randomize the real part with 0 Imaginary part
     *              - Ranging from -100 to 100 by default
     */

    const long max_num = 1e8L;
    double range = upper_bound - lower_bound;
    // Setting seed
    srandom(time(NULL));

    // initialize the vector
    Complex* vec = (Complex*)malloc(N * sizeof(Complex));
    for (unsigned int i = 0; i < N; i++)
        vec[i] = Complex(lower_bound + range * (random() % max_num )/ (max_num + 0.0));

    return vec;
}

bool areVecsEqual(Complex *a, Complex *b, int N, const double tol = 1e-6)
{
    // check if two complex vecs are equal within tolerance level
    for (int i = 0; i < N; i++)
    {
        if (abs(a[i].real() - b[i].real()) > tol ||
           abs(a[i].imag() - b[i].imag()) > tol)
            return false;
    }
    return true;
}

//********************************//
// Matrix - Vector Transformation Methods //

template <typename T>
T* flatten_row_major(T** array2D, int rows, int cols)
{
    /*
     * Flatten the 2D Array into 1D in Row-Major Way
     */
    T* flattened = new T[rows * cols];

    // Copy the values
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            flattened[i * cols + j] = array2D[i][j];

    return flattened;
}

template <typename T>
T* flatten_col_major(T** array2D, int rows, int cols)
{
    /*
     * Flatten the 2D Array into 1D in Column-Major Way
     */
    T* flattened = new T[rows * cols];

    // Copy the values
    for (int j = 0; j < cols; j++)
        for (int i = 0; i < rows; i++)
            flattened[i + j * rows] = array2D[i][j];

    return flattened;
}

template <typename T>
T getEntry_row_major(T* arr, int rows, int cols,
                     int rowIdx, int colIdx)
{
    // Get entry of a matrix that's flatten in a row-major 1D array
    if (rowIdx < 0 || rowIdx >= rows || colIdx < 0 || colIdx >= cols)
        throw out_of_range("Index out of range!");

    return arr[rowIdx * cols + colIdx];
}

template <typename T>
T getEntry_col_major(T* arr, int rows, int cols,
                     int rowIdx, int colIdx)
{
    // Get entry of a matrix that's flatten in a col-major 1D array
    if (rowIdx < 0 || rowIdx >= rows || colIdx < 0 || colIdx >= cols)
        throw out_of_range("Index out of range!");

    return arr[rowIdx + rows * colIdx];
}

// Instantiate the template //
// Complex
template void show_vec<Complex>(Complex*, unsigned int);
template void copy_vec<Complex>(Complex*, Complex*, unsigned int);
template Complex* flatten_row_major<Complex>(Complex**, int, int);
template Complex* flatten_col_major<Complex>(Complex**, int, int);
template Complex getEntry_row_major<Complex>(Complex*, int, int, int, int);
template Complex getEntry_col_major<Complex>(Complex*, int, int, int, int);

// double
template void show_vec<double>(double*, unsigned int);
template void copy_vec<double>(double*, double*, unsigned int);
template double* flatten_row_major<double>(double**, int, int);
template double* flatten_col_major<double>(double**, int, int);
template double getEntry_row_major<double>(double*, int, int, int, int);
template double getEntry_col_major<double>(double*, int, int, int, int);

// int
template void show_vec<int>(int*, unsigned int);
template void copy_vec<int>(int*, int*, unsigned int);
template int* flatten_row_major<int>(int**, int, int);
template int* flatten_col_major<int>(int**, int, int);
template int getEntry_row_major<int>(int*, int, int, int, int);
template int getEntry_col_major<int>(int*, int, int, int, int);

// ***************************//

