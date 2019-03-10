import sys, time, os
import h5py as h5
import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray

#Add Modules from other directories
developerDirectory = '/home/bruno/Desktop/Dropbox/Developer/'
toolsDirectory_1 = developerDirectory + 'pyCUDA/tools/'
volumeRenderDirectory = developerDirectory + "pyCUDA/volumeRender/"
# dataDir = '/media/bruno/hard_drive_1/data/'
# dataDir = '/home/bruno/Desktop/hard_drive_1/data/'
# dataDir = '/home/bruno/Desktop/data/'
# dataDir = '/raid/bruno/data/'
cosmo_dir = '/home/bruno/Desktop/Dropbox/Developer/cosmo_sims/'
toolsDirectory = cosmo_dir + "tools/"
sys.path.extend([toolsDirectory, toolsDirectory_1, volumeRenderDirectory] )
# import volumeRender
from cudaTools import setCudaDevice, getFreeMemory, gpuArray3DtocudaArray, np3DtoCudaArray



def load_snapshot( nSnap ):
  nx = ny = nz = 128
  data_float_1 = np.random.rand( nx, ny, nz)
  data_float_1 = np.random.rand( nx, ny, nz)
  return data_float_1, data_float_2 nx, ny, nz



  def change_float_to_char( data_float ):
    data_float = -256*( data_float - 1)
    data_char = data_float.astype( np.uint8 )
    return data_char

current_snap = 0
data_float_1_1_h, data_float_1_2_h, nx, ny, nz = load_snapshot(  current_snap )
data_float_2_1_h, data_float_2_2_h, nx, ny, nz = load_snapshot(  current_snap )
data_float_2_h, nx, ny, nz = load_snapshot( current_snap + 1 )
data_int_1_1_h = change_float_to_char( data_float_1_1_h )
data_int_1_2_h = change_float_to_char( data_float_1_2_h )
data_int_2_1_h = change_float_to_char( data_float_2_1_h )
data_int_2_2_h = change_float_to_char( data_float_2_2_h )



#initialize pyCUDA context
cudaDevice = setCudaDevice( devN=0, usingAnimation=False )
initialMemory = getFreeMemory( show=True )

data_int_1_1_d = gpuarray.to_gpu( data_int_1_1_h )
data_int_1_2_d = gpuarray.to_gpu( data_int_1_2_h )
data_int_2_1_d = gpuarray.to_gpu( data_int_2_1_h )
data_int_2_2_d = gpuarray.to_gpu( data_int_2_2_h )
# data_int_2_d = gpuarray.to_gpu( data_int_2_h)
# volumeRender.plotData_dArray, copyToScreenArray = gpuArray3DtocudaArray( data_int_1_d, allowSurfaceBind=True )
plotData_dArray_1, copyToScreenArray_1 = gpuArray3DtocudaArray( data_int_1_1_d, allowSurfaceBind=True )
plotData_dArray_2, copyToScreenArray_2 = gpuArray3DtocudaArray( data_int_2_1_d, allowSurfaceBind=True )


def change_snapshot( ):
  global current_snap
  global data_int_1_d, data_int_2_d, data_int_2_h
  current_snap += 1
  data_int_1_d.set( data_int_1_h )
  data_int_2_h = load_snapshot( current_snap )
  data_int_2_d.set( data_int_2_h )
  copyToScreenArray()



#set thread grid for CUDA kernels
block_size_x, block_size_y, block_size_z = 8,8,8   #hardcoded, tune to your needs
gridx = nx // block_size_x + 1 * ( nx % block_size_x != 0 )
gridy = ny // block_size_y + 1 * ( ny % block_size_y != 0 )
gridz = nz // block_size_z + 1 * ( nz % block_size_z != 0 )
block3D = (block_size_x, block_size_y, block_size_z)
grid3D = (gridx, gridy, gridz)


#Read and compile CUDA code
print "\nCompiling CUDA code"
cudaCodeString_raw = open("render3D_cuda.cu", "r").read()
cudaCodeString = cudaCodeString_raw # % { "BLOCK_WIDTH":block2D[0], "BLOCK_HEIGHT":block2D[1], "BLOCK_DEPTH":block2D[2], }
cudaCode = SourceModule(cudaCodeString)
# tex_spins = cudaCode.get_texref('tex_data')
surf_data = cudaCode.get_surfref('surf_data')
interpolationKernel = cudaCode.get_function('data_interpolation')
# ########################################################################
# from pycuda.elementwise import ElementwiseKernel
# ########################################################################
# changeIntToFloat = ElementwiseKernel(arguments="float a, float b, int *input, float *output",
# 			      operation = "output[i] = a*input[i] + b;",
# 			      name = "intToFloat_kernel")
# ########################################################################
# floatToUchar = ElementwiseKernel(arguments="float *input, unsigned char *output",
# 				operation = "output[i] = (unsigned char) ( -255*(input[i]-1));",
# 				name = "floatToUchar_kernel")
########################################################################

n_snapshots = 10
n_frames_per_snapshot = 5
n_frames_total = n_frames_per_snapshot * n_snapshots
frame_number = 0
delta_param = 1.0/n_frames_per_snapshot
interp_param = 0
def stepFunction():
  global frame_number, interp_param
  print frame_number, interp_param
  interpolationKernel( np.float32(interp_param), data_int_1_d, data_int_2_d, grid=grid3D, block=block3D )
  interp_param += delta_param
  frame_number += 1
  if frame_number%n_frames_per_snapshot == 0:
     print 'change Snapshot'
     interp_param = 0



for i in range( n_frames_total ):
  stepFunction()

#
# def get_interp_param( )
#
# t = np.float32( 0.5 )
