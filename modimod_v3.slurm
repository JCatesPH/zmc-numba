#!/bin/bash


#SBATCH -J zmc-mod # name of the job

### set output files
#SBATCH -o ./mod.%A.out #output file 
#SBATCH -e ./mod.%A.err #output error file

### Parameters necessary for GPU computation
#SBATCH --qos gpu
#SBATCH -p gpu
#SBATCH --gres gpu:v100:1

# run application
python ./modimod_v3.py
