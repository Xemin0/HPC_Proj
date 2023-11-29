/*
 * Miscellaneous Utilities for Vectors and Vector Manipulations 
 */


#ifndef LIB_VEC_UTILS_H_
#define LIB_VEC_UTILS_H_

#include <complex>

typedef std::complex<double> Complex;

// ** The Use of Template Requires Explicit Instantiation after Function Definitions ** //
// check the corresponding .cpp file : "../utils/vec_utils.cpp"

template <typename T>
void show_vec(T *vec, unsigned int N);

template <typename T>
void copy_vec(T *vec1, T *vec2, unsigned int N);
// Copy vec1 into vec2

Complex* rand_vec(unsigned int N, double lower_bound = -100, double upper_bound = 100);
/*
 * Generate a random N dimensional complex vector
 *
 * ## Currently:
 *              - Only randomize the real part with 0 Imaginary part
 *              - Ranging from -100 to 100 by default
 */


bool areVecsEqual(Complex *a, Complex *b, int N, const double tol = 1e-6);
// check if two complex vecs are equal within the given tolerance level


//********************************//
// Matrix - Vector Transformation Methods //

template <typename T>
T* flatten_row_major(T** array2D, int rows, int cols);

template <typename T>
T* flatten_col_major(T** array2D, int rows, int cols);

template <typename T>
T getEntry_row_major(T* mat, int rows, int cols,
                     int rowIdx, int colIdx);

template <typename T>
T getEntry_col_major(T* mat, int rows, int cols,
                     int rowIdx, int colIdx);


#endif /* LIB_VEC_UTILS_H_*/
