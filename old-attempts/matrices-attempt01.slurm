#!/bin/bash

#SBATCH -J zmc-ds-func-02 # name of the job

### set output files
#SBATCH -o ./ds-output/zmc-matrices.%A.out #output file 
#SBATCH -e ./ds-output/zmc-matrices.%A.err #output error file

### Parameters necessary for GPU computation
#SBATCH --qos gpu
#SBATCH -p gpu
#SBATCH --gres gpu:v100:1

# run application
srun python ./matrices-attempt-02.py
