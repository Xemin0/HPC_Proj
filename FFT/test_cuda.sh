#!/bin/bash

# Job name
#SBATCH -J FFT_CUDA

# Request a GPU node and 1 GPU
#SBATCH -p gpu --gres=gpu:1

# Ensures all allocated cores are on the same node
#SBATCH -N 1

# Request 1 CPU core
#SBATCH -n 1

# Request Memory
#SBATCH --mem=25G

#SBATCH -t 00:20:00

#SBATCH -e ./Data/Oscar/job-%J.err
#SBATCH -o ./Data/Oscar/job-%J.out

# ==== End of SBATCH settings ==== #
# Check GPU info
nvidia-smi

# Load CUDA and gcc on Oscar
module load cuda/11.2.0 gcc/10.2 fftw/3.3.6

# Compile
make all

# Run and Profile
#nsys profile --stats=true --force-overwrite=true -o report1 ./It_CT.out
./It_CT.out
