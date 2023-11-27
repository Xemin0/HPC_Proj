/*
 * 1-D FFT with Iterative Cooley-Tukey Algorithm
 * 
 */

#include <iostream>
#include <complex>
#include <stdlib.h>
#include <cmath>
#include <fftw3.h>

#include <time.h> // time as the seed for rand()
#include <sys/time.h> // for gettimeofday() sys call
using namespace std;

const double PI = 3.141592653589793238460;
typedef std::complex<double> Complex;

// ** This Method Overwrites the Input ** //
// ** Remember to create a copy of the input ** //

//bit reverse
void bitReverse(Complex *x, int N)
{
    for (int i = 1, j = 0; i < N; i++)
    {
        int bit = N >> 1;
        for (; j&bit; bit >>= 1)
        {
            j ^= bit;
        }
        j ^= bit;
        if (i < j)
            std::swap(x[i], x[j]);
    }
}

// Iterative 1D FFT 
void fft(Complex *x, int N)
{
    bitReverse(x, N);
    for (int len = 2; len <= N; len <<= 1)
    {
        double angle = -2*PI/len;
        Complex wlen(cos(angle), sin(angle));
        for (int i = 0; i < N; i += len)
        {
            Complex w(1);
            for (int j = 0; j < len/2; j++)
            {
                Complex u = x[i+j];
                Complex v = x[i + j + len/2]*w;
                x[i+j] = u + v;
                x[i + j + len/2] = u - v;
                w *= wlen;
            }
        }
    }
}



// *******************
// Misc Utilities
void show_vec(Complex *vec, unsigned int N)
{
    for (unsigned int i = 0; i < N; i++)
        cout << *(vec + i) << " ";

    cout << endl;
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

void copy_vec(Complex *vec1, Complex *vec2, unsigned int N)
{
    // Copy vec1 into vec2
    for (int i = 0; i < N; i++)
        vec2[i] = vec1[i];
}

void fftw2Complex(fftw_complex *arr, Complex *x, unsigned int N)
{
    // fftw_complex to Complex dtype
    for (int i = 0; i < N; i++)
    {
        x[i] = Complex(arr[i][0], arr[i][1]);
    }
}

void Complex2fftw(Complex *x, fftw_complex *arr, unsigned int N)
{
    // Complex to fftw_complex dtype
    for (int i = 0; i < N; i++)
    {
        arr[i][0] = x[i].real();
        arr[i][1] = x[i].imag();
    }
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
// *******************

// Performance
unsigned long get_time(){
    // micro-second (us)
    struct timeval curr_time;
    gettimeofday(&curr_time, 0); // get sys time 

    return (curr_time.tv_sec * 1e6) + curr_time.tv_usec; // Combining the two returned elements in microseconds
}

// Main Driver for Testing
int main(int argc, char** argv)
{
    // Check for correct number of arguments
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " fft_type print_option\n";
        return 1;
    }

    // Parse arguments
    string fft_type = argv[1]; // "custom" or "fftw" or "both"
    string print_option = argv[2]; // "yes" or "no"

    unsigned int N = 16;
    // randomize a vector
    // vec      : base vec
    // vec_fftw : used to store results from FFTW
    Complex *vec = rand_vec(N);
    Complex *vec_fftw = rand_vec(N);

    // Creating a Copy of vec 
    // used by our method
    Complex *vec_cpy = (Complex*)malloc(N * sizeof(Complex));
    copy_vec(vec, vec_cpy, N);


    if (fft_type == "custom") {
        // Run custom FFT
        fft(vec_cpy, N);

        if (print_option == "yes") {
            //cout << "FFT result by CT Method:" << endl;
            show_vec(vec_cpy, N);
        }
    } 
    
    else if (fft_type == "fftw") {
        // Run FFTW
        fftw_complex vec_in[N], vec_out[N];
        Complex2fftw(vec, vec_in, N);

        fftw_plan p = fftw_plan_dft_1d(N, vec_in, vec_out, FFTW_FORWARD, FFTW_ESTIMATE);
        fftw_execute(p);

        Complex *vec_fftw = (Complex*)malloc(N * sizeof(Complex));
        fftw2Complex(vec_out, vec_fftw, N);

        if (print_option == "yes") {
            //cout << "FFT result by standard library:" << endl;
            show_vec(vec_fftw, N);
        }

        fftw_destroy_plan(p);
        free(vec_fftw);
    } 
    
    else if (fft_type == "both") {
        // ####### FFT standard
        // create input and output vectors for FFTW using stack allocation
        fftw_complex vec_in[N], vec_out[N];
        Complex2fftw(vec, vec_in, N);

        // Creating plan for 1D FFT in FFTW
        fftw_plan p;
        p = fftw_plan_dft_1d(N, vec_in, vec_out, FFTW_FORWARD, FFTW_ESTIMATE);
        fftw_execute(p); 

        // Convert to Complex dtype
        fftw2Complex(vec_out, vec_fftw, N);

        fftw_destroy_plan(p);

        // ####### FFT CT Method
        fft(vec_cpy, N);

        // Show both vecs
        //cout << "FFT result by standard library:" << endl;
        show_vec(vec_fftw, N);

        //cout << "FFT result by CT Method:" << endl;
        show_vec(vec_cpy, N);
    } 
    
    else {
        std::cerr << "Invalid FFT type. Use 'custom' or 'fftw'.\n";
        return 1;
    }

    // Check if two vecs are equal 
    // cout << boolalpha;
    // cout << "Are FFT results the same?: " << bool(areVecsEqual(vec_fftw, vec_cpy, N)) << endl;
    
    // free memory
    free(vec);
    free(vec_cpy);
    free(vec_fftw);

    //fftw_free(vec_in);
    //fftw_free(vec_out); // no need to reclaim memory for stack allocation
}
