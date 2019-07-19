#!/bin/bash


#SBATCH -J linalg-test # name of the job

### set output files
#SBATCH -o ./linalg.%A.out #output file 
#SBATCH -e ./linalg.%A.err #output error file

### Parameters necessary for GPU computation
#SBATCH --qos gpu
#SBATCH -p gpu
#SBATCH --gres gpu:v100:1

# run application
python ./linearalg-cuda.py
