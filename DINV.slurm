#!/bin/bash


#SBATCH -J DINV # name of the job

### set output files
#SBATCH -o ./DINV.%A.out #output file 
#SBATCH -e ./DINV.%A.err #output error file

### Parameters necessary for GPU computation
#SBATCH --qos gpu
#SBATCH -p gpu
#SBATCH --gres gpu:v100:1

# run application
srun python ./DINV.py
