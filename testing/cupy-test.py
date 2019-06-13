import numba
from numba import cuda
import numpy
import cupy

a = cupy.random.randn(1, 2, 3, 4).astype(cupy.float32)
result = numpy.sum(a)

print('Sum of vector a:', result)


c = cupy.arange(3, dtype=cupy.float)
print('\n')
print(c)


d = cupy.identity(3, dtype=cupy.float32)
e = cupy.asnumpy(d)
print('\n')
print(e)

vec = [1,1,1]
f = cupy.diag(vec,k=1)

print('\n')
print(f)

@cuda.jit(device=True)
def devicefoo(a,b):
	c = 
	return d
	

@cuda.jit()
def cudafoo(x, y, N, z):
	tid = cuda.threadIdx.x
	blkid = cuda.blockIdx.x
	blkdim = cuda.blockDim.x
	
	i = tid + blkid * blkdim

	if i >= N:
		return
	
	z[i] = devicefoo(x,y)
	

N=3
#v = np.array([2,2,2])
y = cupy.ones(N)
x = cupy.array([10,20,30])
z = cupy.zeros(N)
cudafoo[32,1](x, y, N, z)
print('\ny=', y)

print('\nx=', x)

print('\nz=', z)

