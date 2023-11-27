/*
 * Miscellaneous Utilities for Vectors and Vector Manipulations 
 */


#ifndef LIB_VEC_UTILS_H_
#define LIB_VEC_UTILS_H_

#include <complex>

typedef std::complex<double> Complex;

void show_vec(Complex *vec, unsigned int N);

void copy_vec(Complex *vec1, Complex *vec2, unsigned int N);
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

#endif /* LIB_VEC_UTILS_H_*/
