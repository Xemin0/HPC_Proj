/*
 * Time measurements on CPU and GPU 
 * (requires `nvcc` for any .cu that includes this)
 *
 */

#ifndef LIB_TIMER_H_
#define LIB_TIMER_H_


#include <chrono> // CPU High Precision Timer
//#include <cuda.h> // GPU High Precision Timer


// ### CPU Timer ####
unsigned long get_time(); // microsecond (us)


// ### GPU Timer ####
//struct gpuTime;

// ### Timer for Both CPU and GPU ### //
struct HighPrecisionTimer
{
    /*
     * High Precision Timer in microsecond(us)
     * that provide high precision time measurement on both CPU and GPU 
     */
    std::chrono::time_point<std::chrono::high_resolution_clock> cpu_start;
    std::chrono::time_point<std::chrono::high_resolution_clock> cpu_stop; // CPU

    //cudaEvent_t gpu_start;
    //cudaEvent_t gpu_stop; // GPU

    // Constructor
    HighPrecisionTimer()
    {   
        //cudaEventCreate(&gpu_start);
        //cudaEventCreate(&gpu_stop); // GPU
    }   

    // Destructor
    ~HighPrecisionTimer()
    {   
        //cudaEventDestroy(gpu_start);
        //cudaEventDestroy(gpu_stop); // GPU
    }   

    void Start()
    {   
        cpu_start = std::chrono::high_resolution_clock::now();
        //cudaEventRecord(gpu_start, 0); 
    }   

    void Stop()
    {   
        cpu_stop = std::chrono::high_resolution_clock::now();
        //cudaEventRecord(gpu_stop, 0); 
    }   

    float Elapsed(bool isCPU = true) // microsecond (us)
    {   
        float elapsed = -1.0;
        if (isCPU){ // CPU time
           elapsed = std::chrono::duration<float, std::micro>(cpu_stop - cpu_start).count();
        }
        /*
        else{ // GPU time
            cudaEventSynchronize(gpu_stop);
            cudaEventElapsedTime(&elapsed, gpu_start, gpu_stop);

            elapsed *= 1000.0;
        }
        */
        return elapsed;
    }   
};


#endif /* LIB_TIMER_H_ */
