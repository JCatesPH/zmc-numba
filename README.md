# zmc-numba

This project uses the ZMCintegral package with the numba interface, which is available at https://github.com/Letianwu/ZMCintegral .

All calculations are done on the University of Alabama's HPC with a NVIDIA Tesla P100 with 16G of memory.

# Conda requirements:
  
  - numba
  - llvmlite
  - cudatoolkit
  - ZMCintegral
  - numpy
  - scipy
  
  
  `conda install numba cudatoolkit llvmlite numpy scipy -c zhang-junjie ZMCintegral`
