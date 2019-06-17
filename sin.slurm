#!/bin/bash


#SBATCH -J zmc-sin # name of the job

### set output files
#SBATCH -o ./sin.%A.out #output file 
#SBATCH -e ./sin.%A.err #output error file

### Parameters necessary for GPU computation
#SBATCH --qos gpu
#SBATCH -p gpu
#SBATCH --gres gpu:v100:1

# run application
srun python ./sin-test.py
