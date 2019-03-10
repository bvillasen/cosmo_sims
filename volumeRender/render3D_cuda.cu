// #include <pycuda-complex.hpp>
#include <surface_functions.h>
#include <stdint.h>
#include <cuda.h>

typedef unsigned char uchar;

surface< void, cudaSurfaceType3D> surf_data;

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

__global__ void data_interpolation( float t, uchar *data_1, uchar *data_2 ){
  int nx = blockDim.x * gridDim.x;
  int ny = blockDim.y * gridDim.y;
  int t_j = blockIdx.x*blockDim.x + threadIdx.x;
  int t_i = blockIdx.y*blockDim.y + threadIdx.y;
  int t_k = blockIdx.z*blockDim.z + threadIdx.z;
  int tid = t_j + t_i*nx + t_k*nx*ny;

  uchar value_1 = data_1[tid];
  uchar value_2 = data_2[tid];

  uchar value = 128;

  surf3Dwrite(  value, surf_data,  t_j*sizeof(uchar), t_i, t_k,  cudaBoundaryModeClamp);


}
