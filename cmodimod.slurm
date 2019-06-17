#!/bin/bash


#SBATCH -J zmc-cmod # name of the job

### set output files
#SBATCH -o ./cmod.%A.out #output file 
#SBATCH -e ./cmod.%A.err #output error file

### Parameters necessary for GPU computation
#SBATCH --qos gpu
#SBATCH -p gpu
#SBATCH --gres gpu:v100:1

# run application
srun python ./cmodimod.py
