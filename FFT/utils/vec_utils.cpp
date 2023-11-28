/*
 * Miscellaneous Utilities for Vectors and Vector Manipulations 
 */



#include <complex>
#include <stdlib.h>
#include <time.h> // seed for random generator
#include <cmath>
#include <iostream>
using namespace std;

typedef std::complex<double> Complex;

void show_vec(Complex *vec, unsigned int N)
{
    for (unsigned int i = 0; i < N; i++)
        cout << *(vec + i) << " ";

    cout << endl;
}


void copy_vec(Complex *vec1, Complex *vec2, unsigned int N)
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
