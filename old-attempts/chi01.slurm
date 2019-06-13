#!/bin/bash


#SBATCH -J zmc-chi # name of the job

### set output files
#SBATCH -o ./ds-output/chi.%A.out #output file 
#SBATCH -e ./ds-output/chi.%A.err #output error file

### Parameters necessary for GPU computation
#SBATCH --qos gpu
#SBATCH -p gpu
#SBATCH --gres gpu:v100:1

# run application
srun python ./chi-01.py
