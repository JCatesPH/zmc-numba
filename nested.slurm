#!/bin/bash


#SBATCH -J zmc-nested # name of the job

### set output files
#SBATCH -o ./nested.%A.out #output file 
#SBATCH -e ./nested.%A.err #output error file

### Parameters necessary for GPU computation
#SBATCH --qos gpu
#SBATCH -p gpu
#SBATCH --gres gpu:v100:1

# run application
srun python ./nested-loop.py
