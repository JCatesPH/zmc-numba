#!/bin/bash

#SBATCH -J zmc-chi-N # name of the job

### set output files
#SBATCH -o ./chiN.%A.out #output file 
#SBATCH -e ./chiN.%A.err #output error file

### Parameters necessary for GPU computation
#SBATCH --qos gpu
#SBATCH -p gpu
#SBATCH --gres gpu:v100:1

# run application
srun python ./chi-N3-v4.py
