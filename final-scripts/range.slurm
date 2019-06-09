#!/bin/bash


#SBATCH -J zmc-range # name of the job

### set output files
#SBATCH -o ./range.%A.out #output file 
#SBATCH -e ./range.%A.err #output error file

### Parameters necessary for GPU computation
#SBATCH --qos gpu
#SBATCH -p gpu
#SBATCH --gres gpu:v100:1

# run application
srun python ./range-mod.py
