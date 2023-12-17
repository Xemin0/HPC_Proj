/*
 * Cuda Device Functions for Math Operations
 *
 */

#include <cuda.h>
#include <cuComplex.h>


// Power Function for Complex
__device__ cuDoubleComplex pow_cuDoubleComplex(cuDoubleComplex z, int n){
    double r = cuCabs(z); // Magnitude
    double theta = atan2(cuCimag(z), cuCreal(z)); // Argument

    double rn = pow(r, n);
    double nTheta = n * theta;

    return make_cuDoubleComplex(rn * cos(nTheta), rn * sin(nTheta));
}
