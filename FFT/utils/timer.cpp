/*
 * Time measurements on CPU 
 * (and GPU; require CUDA and `nvcc` with `.cu` as the file extension)
 *
 */

#include <sys/time.h> // for gettimeofday() sys call  


// ### CPU Timer ####
unsigned long get_time(){
    // micro-second (us)
    struct timeval curr_time;
    gettimeofday(&curr_time, 0); // get sys time 

    return (curr_time.tv_sec * 1e6) + curr_time.tv_usec; // Combining the two returned elements in microseconds
}

