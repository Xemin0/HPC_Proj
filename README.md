# Parallelizing 2D Fast-Fourier Transform Implemented with Cooley-Tukey Algorithm Using CUDA

### Results on Current Implementation of 1D FFT for the Given 1D Dataset (Updated; *Need Further Optimizations*)
|1D FFT method| Performance (ms)|
|:---:|:---:|
|FFTW(CPU)| 396.52 |
|Iterative Method(CPU)| 619.40 |
|Method with Batch Input<br />(GPU - 500 Thread Blocks)| 25.29|
|Method with Streams<br />(GPU - 5 Streams & 100 Thread Blocks per Stream)|19.44|

### APIs Proportion of Current Implementations
<!--
![CUDA APIs Proportion](./results/pix/APIcallsProportion.png?raw=true "NSIGHT Profile") 
-->
|Method| APIs Proportion of Current Implementations|
|:---:| :---:|
|500 ThreadBlocks|<img src="./results/pix/APIcallsProportion.png" width="600">|
|5 Streams & 100 Thread Blocks per Stream|<img src="./results/pix/APIproportionStreams.png" width="600">|


### Datasets
- [1D Test Dataset(EEG Data)](https://drive.google.com/drive/folders/1HtinENxel10tj7W3ckKxAQix1gxDz1br?usp=sharing)
- [2D Test Dataset(CIFAR10)](https://drive.google.com/drive/folders/1qvbv90PAO79KGgIBNRjyjQkqEzqHD7wI?usp=sharing)

## Decision Making in Parallelizing an Algorithm
```mermaid
flowchart TD
	N[Rewrite the Algorithm] --> A
	A --> |Def NO| N
	subgraph code [KNOW UR CODE]
	N
	A
	end
	A{is Parallelizable?}
	Y[["Distribution of Work
	Memory Access Pattern
	Occupancy/Device Usage"]]
	A -->|Independency-YES| Y
	A --> |Dependencies-Maybe NO?| M{{"Requires
	 Synchronizations
	 Barriers"}}
	subgraph machine [KNOW UR MACHINE]
	Y
	M
	end

    style Y fill:#f6f,stroke:#333,stroke-width:4px
	style M fill:#f6f,stroke:#333,stroke-width:4px
	style A fill:#99d,stroke:#f66,stroke-width:2px
```

## File Structure
**Each folder has (and will have) its respective `README` describing what type of modules should be placed inside**

**Make sure you have the following folders**
- Data Files should be placed in `./FFT/Data/` 
- Output results by methods from `/FFT/eval/` will be output to `/FFT/Data/Results/`
- `test_cuda.sh` will output buffered printouts in `FFT/Data/Oscar/`

## Prerequisite
- [FFTW C/C++ library](https://www.fftw.org/) for validating the results (not necessary)
    - `MacOS` installation: `brew install fftw`
- Most likely a 'Turing Arch' GPU with `CUDA11.2.0` 

## Compile with `FFTW` library
*Change the `FFTW` library path as needed*

### Compile Using the `makefile` (recommended)
In the `./FFT/` folder use the bash command `make`

**May need to change `NVCCFLAG` for GPUs with different Compute Capabilities (CC), but no CC-specific CUDA API calls in current implementation**

*If a fresh build is desired, simply run `make clean` before running `make` or `make all`*

## Run Locally 
In the `./FFT/` folder run `./It_CT.out`

or `nsys profile ./It_CT.out` to profile with NVIDIA NSIGHT and run

## Compile and Run with SLURM Script on a Cluster
In the `./FFT/` folder use the SLURM `test_cuda.sh` on Oscar by 
```bash
sbatch test_cuda.sh
```

## Major Challenges
### 1D FFT
- Precision Concerns: Manipulation of Complex numbers; Cooley-Tukey method has runtime $O(N \log N)$, meaning the (primarily truncation) errors will build up at least proportional to this quantity
- Memory Management on `DEVICE` for FFT on Longer Vectors: currently using shared memory for data exchange between threads, but the memory size scales with problem size $N$ for this implementation (Limitation on Shared Memory size for each BLOCK/SM is Compute-Capability(CC) specific; GPU arch prior to `Volta` with CC 7.0 is limited to 48KB per BLOCK/SM, while `Double Precision Complex` takes 16 Bytes); primarily two options:
    - A Combination of Shared Mermoy and Warp-Level Primitives: requires a sophisticated design because a changing number of threads are dynamically assigned for each sub-vector in each outer for loop iteration
    - Stream-Lining the Outer for Loop in the Current Implementation of C-T Algorithm: Sequential Kernel Calls instead of just one  
    - Breaking Down/Down-Sampling the long vector to smaller subvectors; difficulties involving **Fourier Theories**
        - Frequency Leakage
        - Gibbs Effects
        - Aliasing (Nyquist Sampling Theorem)
- Plausibility of Nested Kernel Calls??

### 2D FFT
A couple implementation details to consider
- For effective data exchanges among a 2D thread BLOCK, may need to implement 2D FFT from scratch without sequential subKernel Calls
- Alternatively, offload row-wise 1D FFT calls to CUDA Streams before writing to the global memory for later column-wise 1D FFTs
- Or utilize 1D Batch FFT

### Performance Evaluation Workflow for CUDA Methods
- As `cudaMalloc` and `cudaFree` take up **RIDICULOUS** amount of time compared to kernels, for larger datasets, consider processing data as a batch.
    - Larger in Vector/Matrix Sizes
    - Larger in the Number of Samples



## TO-DOS and Progress
### 1D FFT
- ~Correctness of 1D FFT CUDA method~
- Optimization for long vector FFT
- 1D FFT for batch input
    - ~Sequentially launch the kernels~
    - ~Using thread blocks to process the input in batches(Tiling)~
        - Test and plot the relationship between performance and number of thread blocks assigned; Compare with theoretical value
        - Add boundary checks when tiling in 1D Batch FFT
        - Correctness
    - ~Streams with asynchronous operations~
        - Test and plot the relationship between performance and number of streams
        - Correctness
    - ~Implement `eval_performance` for batched methods~
- **Implement 1D FFT for arbitrary input sizes**

### 2D FFT
- ?? Design Kernels for 2D FFT (Nested Calls?)
- ?? Directly Optimize 2D FFT after `bitReverse` step
- Launch 1D FFT for batch input twice
- Correctness and Performance

### Misc
- ~`cudaCheck()`~
- ~Separate CUDA Utilities~
- ~Add data processing and memory allocation utils for CUDA~

### Primary Files to Work On
- `CooleyTukeyFFT/*.cu`
- `eval/eval_correctness.cu`
- `eval/eval_performance.cu`
- Corresponding Headers
- ~Makefile~
- ~`SLURM` script~
