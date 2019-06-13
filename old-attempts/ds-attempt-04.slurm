#!/bin/bash


#SBATCH -J zmc-ds-func-04 # name of the job

### set output files
#SBATCH -o ./ds-output/ds-integral.%A.out #output file 
#SBATCH -e ./ds-output/ds-integral.%A.err #output error file

### Parameters necessary for GPU computation
#SBATCH --qos gpu
#SBATCH -p gpu
#SBATCH --gres gpu:v100:1

# run application
srun python ./ds-attempt-04.py
