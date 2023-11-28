/*
 * Time measurements on CPU 
 * (and GPU; require CUDA and `nvcc` with `.cu` as the file extension)
 *
 */

#ifndef LIB_TIMER_H_
#define LIB_TIMER_H_

// ### CPU Timer ####
unsigned long get_time(); // microsecond (us)


// ### GPU Timer ####
//struct gpuTime;


#endif /* LIB_TIMER_H_ */
