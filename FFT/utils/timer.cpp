/*
 * Time measurements on CPU 
 * (and GPU; require CUDA and `nvcc` with `.cu` as the file extension)
 *
 */

#include <sys/time.h> // for gettimeofday() sys call  

//#include <cuda.h>

// ### CPU Timer ####
unsigned long get_time(){
    // micro-second (us)
    struct timeval curr_time;
    gettimeofday(&curr_time, 0); // get sys time 

    return (curr_time.tv_sec * 1e6) + curr_time.tv_usec; // Combining the two returned elements in microseconds
}



// ### GPU Timer ####
/*
struct gpuTime
{
    cudaEvent_t start;
    cudaEvent_t stop;

    gpuTimer()
    {   
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }   

    ~gpuTimer()
    {   
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }   

    void Start()
    {   
        cudaEventRecord(start, 0); 
    }   

    void Stop()
    {   
        cudaEventRecord(stop, 0); 
    }   

    float Elapsed() // millisecond (ms)
    {   
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }   
};
*/

