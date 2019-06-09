#!/bin/bash


#SBATCH -J zmc-chi-N # name of the job

### set output files
#SBATCH -o ./chi.%A.out #output file 
#SBATCH -e ./chi.%A.err #output error file

### Parameters necessary for GPU computation
#SBATCH --qos gpu
#SBATCH -p gpu
#SBATCH --gres gpu:v100:1

# run application
srun python ./chi-N1-v5.py
