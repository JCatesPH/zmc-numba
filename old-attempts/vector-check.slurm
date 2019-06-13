#!/bin/bash

#SBATCH -J ds-vectorized # name of the job

### set output files
#SBATCH -o ds-vector.%A.out #output file 
#SBATCH -e ds-vector.%A.err #output error file

### Parameters necessary for GPU computation
#SBATCH --qos gpu
#SBATCH -p gpu
#SBATCH --gres gpu:v100:1

# run application
srun python ./check-vectorized.py
