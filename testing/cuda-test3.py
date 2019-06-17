import numba

from numba import cuda

import numpy as np

import pyculib as pc

 

A = np.random.rand(3,3)

#Create Random 3X3 matrix

 

B = np.random.rand(3,3)  

#Create Random 3X3 matrix

C = np.random.rand(3,3)  

#Create Random 3X3 matrix

 

Bg = numba.cuda.to_device(B )

#Transfer to GPU

Ag = numba.cuda.to_device(A)

 #Transfer to GPU

Cg=numba.cuda.to_device(C)

#Transfer to GPU

 

 

blOb = pc.blas.Blas()

 

 

yy=blOb.gemm('N','N',3,3,3,1,Ag,Bg,0,Cg)

#Here ‘N’ is normal (not transpose)

Y=Cg.copy_to_host(C)  

#Transfer to CPU

 

print(Y)

#Print Result

print(np.matmul(A,B))

#compare to NumPy