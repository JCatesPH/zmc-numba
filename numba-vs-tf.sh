#!/bin/bash

# TAKING FILES TO COMPARE FROM THE COMMAND LINE
FILE1=$1
FILE2=$2

# CHANGE TO THE PROPER ENVNAME HERE
ENV1="zmcint"
ENV2="tf"

echo "======================================================================="
echo "Comparing the time of two python scripts"
echo "======================================================================="
echo -n "This is the case of "
echo -n $1
echo -n "vs"
echo -n $2
echo "======================================================================="
echo $1

# SETTING ENVIRONMENT
conda deactivate
source activate $ENV1

time python $FILE1

echo "Script 1 is finished."
echo "======================================================================="
echo $1

# SETTING ENVIRONMENT
conda deactivate
source activate $ENV2

time python $FILE2

echo "Script 2 is finished."