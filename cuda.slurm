#!/bin/bash


#SBATCH -J zmc-cuda # name of the job

### set output files
#SBATCH -o ./cuda.%A.out #output file 
#SBATCH -e ./cuda.%A.err #output error file

### Parameters necessary for GPU computation
#SBATCH --qos gpu
#SBATCH -p gpu
#SBATCH --gres gpu:v100:1

# run application
srun python ./cuda-test.py
