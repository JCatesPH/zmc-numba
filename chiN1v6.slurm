#!/bin/bash


#SBATCH -J zmc-chi # name of the job

### set output files
#SBATCH -o ./chiN1v6.%A.out #output file 
#SBATCH -e ./chiN1v6.%A.err #output error file

### Parameters necessary for GPU computation
#SBATCH --qos gpu
#SBATCH -p gpu
#SBATCH --gres gpu:v100:1

# run application
srun python ./chi-N1_v6.py
