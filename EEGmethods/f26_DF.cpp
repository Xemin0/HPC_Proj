/*
 * Function 26: Dominat Frequencies
 *
 * Method: For each channel calculate the Power Spectral Density (PSD):
 *         Using `pwelch` method with a sampling frequency
 *         Choose the fequency with the highest power for each channel 
 * 
 * Input: 
 *       - X : n by m matrix
 *             n channels; m timesteps
 *       
 * Output:
 *       -DF : 1 by n vector of dominant frequencies for each channel
 */

#include <iostream>
//#include <stdio.h>

using namespace std;

struct PSD{
    float *pxx;
    float *freq;
};

// Forward Declaration of pwelch (PSD)
PSD pwelch(float *Xi, unsigned int m, unsigned int fs = 200);
/*
 * Xi: Vector; One channel of a time series
 * m : Length of timeseris Xi
 * fs: Positive int; Sampling frequency
 */


float* dominant_frequencies(float **X, 
                            unsigned int n, 
                            unsigned int m,
                            unsigned int fs = 200)
{
    // Initialize output dominant frequencies
    float *DF = new float[n];
    // Initialize PSD for each channel;
    PSD *psd = new PSD[n];

    for (int i = 0; i < n; ++i)
    {
       psd[i] = pwelch(X[i], m, fs);
       // Finding the index of max in resulting pxx
       unsigned int max_idx = distance(psd[i].pxx, 
                                       max_element(psd[i].pxx, psd[i].pxx + m));
       // get the frequency for the highest power
       DF[i] = psd[i].freq[max_idx];  
    }
    return DF; 
}
