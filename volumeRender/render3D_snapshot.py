import sys, time, os
from os import listdir
from os.path import isfile, join
import h5py as h5
import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import matplotlib.colors as cl
import matplotlib.cm as cm
# import sys, pygame
# pygame.init()

currentDirectory = os.getcwd()
#Add Modules from other directories
developerDirectory = '/home/bruno/Desktop/Dropbox/Developer/'
toolsDirectory_1 = developerDirectory + 'pyCUDA/tools/'
volumeRenderDirectory = developerDirectory + "pyCUDA/volumeRender/"
# dataDir = '/media/bruno/hard_drive_1/data/'
# dataDir = '/home/bruno/Desktop/hard_drive_1/data/'
dataDir = '/home/bruno/Desktop/data/'
# dataDir = '/raid/bruno/data/'
cosmo_dir = '/home/bruno/Desktop/Dropbox/Developer/cosmo_sims/'
toolsDirectory = cosmo_dir + "tools/"
sys.path.extend([toolsDirectory, toolsDirectory_1, volumeRenderDirectory] )
import volumeRender_new
from cudaTools import setCudaDevice, getFreeMemory, gpuArray3DtocudaArray, np3DtoCudaArray
from load_data_cholla import load_snapshot_data

nFields = 1
transp_type = 'sigmoid'
useDevice = 0
for option in sys.argv:
  if option == "anim": usingAnimation = True
  if option.find("dev=") != -1: useDevice = int(option[-1])
  if option.find("n=") != -1: nFields = int(option[-1])
  if option.find("sigmoid") != -1: transp_type = 'sigmoid'
  if option.find("iso") != -1: transp_type = 'gaussian'


# inDir = dataDir + 'cholla_hydro/collapse_3D/'
inDir = dataDir + 'cosmo_sims/cholla_pm/cosmo_256_hydro/set_1/'
outDir = '/home/bruno/Desktop/anim_cosmo/'

# #Cholla fie names
partFileName = inDir + 'data_particles.h5'
gridFileName = inDir + 'data_grid.h5'


dataFiles = [f for f in listdir(inDir) if (isfile(join(inDir, f)) and (f.find('.h5') > 0 ) and ( f.find('_') < 0) ) ]
dataFiles = np.sort( dataFiles )
nFiles = len( dataFiles )
nSnapshots = nFiles


def prepare_data( nSnap, plotData, max_vals=[], log=False ):
  if log : plotData = np.log10(plotData + 1)
  # plotData -= plotData.min()
  # max_val = max_vals[nSnap]
  # norm_val = max_vals[nSnap:nSnap+5].mean()
  # norm_val = max_vals[-1]
  norm_val = plotData.max()
  plotData /= norm_val
  plotData_h_256 = (-255*(plotData-1)).astype(np.uint8)
  n=3
  plotData_h_256[:,:n,:n] = 0
  plotData_h_256[:n,:,:n] = 0
  plotData_h_256[:n,:n,:] = 0
  plotData_h_256[-n:,-n:,:] = 0
  plotData_h_256[:,-n:,-n:] = 0
  plotData_h_256[-n:,:,-n:] = 0
  plotData_h_256[-n:,:n,:] = 0
  plotData_h_256[-n:,:,:n] = 0
  plotData_h_256[:,:n,-n:] = 0
  plotData_h_256[:,-n:,:n] = 0
  plotData_h_256[:n,:,-n:] = 0
  plotData_h_256[:n,-n:,:] = 0
  return plotData_h_256

# nSnap = 9
# inFileName = inDir + 'data_grid.h5'
# inFile = h5.File( inFileName, 'r')
#
# def load_data_snap( nSnap, inFile ):
#   snapKey = str(nSnap)
#   snapData = inFile[snapKey]
#   density = snapData['density'][...]
#   return density
#
# data = load_data_snap( nSnap, inFile )

# def load_data_snap( nSnap):
#   inFileName = inDir + '{0}.h5'.format( nSnap)
#   inFile = h5.File( inFileName, 'r')
#   density = inFile['density'][...]
#   # energy = inFile['Energy'][...]
#   data = density
#   plotData_h_256 = prepare_data( nSnap, data, log=False)
#   return plotData_h_256


def load_data_snap( nSnap):
  snapKey = str( nSnap )
  data_cholla, nSnapshots = load_snapshot_data( snapKey, gridFileName, partFileName )
  density_dm = data_cholla['dm']['density'][...]
  plotData_h_256 = prepare_data( nSnap, density_dm, log=False)
  return plotData_h_256

def change_snapshot( nSnap):
  global plotData_h_256, copyToScreenArray
  new_data = load_data_snap( nSnap )
  plotData_h_256 = new_data.copy()
  # copyToScreenArray()
  volumeRender.plotData_dArray, copyToScreenArray = np3DtoCudaArray( plotData_h_256 )


nSnap = 0
plotData_h_256 = load_data_snap( nSnap )
nz, ny, nx = plotData_h_256.shape
nWidth, nHeight, nDepth = nx, ny, nz

#Initialize openGL
if nFields == 2: volumeRender.nTextures = 2
volumeRender.transp_type = transp_type
volumeRender.colorMap = 'CMRmap'
volumeRender.trans_ramp_0 = np.float32( 7.5 )
volumeRender.trans_center_0 = np.float32( 0.15 )
volumeRender.trans_ramp_1 = np.float32( 7.5 )
volumeRender.trans_center_1 = np.float32( 0.15 )
if transp_type == 'gaussian':
  volumeRender.trans_ramp_0 = np.float32( 0.1 )
  volumeRender.trans_center_0 = np.float32( 0.3 )

volumeRender.density = np.float32(0.1)
volumeRender.nWidth = nWidth
volumeRender.nHeight = nHeight
volumeRender.nDepth = nDepth
volumeRender.windowTitle = "Cosmo Volume Render"
volumeRender.initGL()

#set thread grid for CUDA kernels
block_size_x, block_size_y, block_size_z = 8,8,8   #hardcoded, tune to your needs
gridx = nWidth // block_size_x + 1 * ( nWidth % block_size_x != 0 )
gridy = nHeight // block_size_y + 1 * ( nHeight % block_size_y != 0 )
gridz = nDepth // block_size_z + 1 * ( nDepth % block_size_z != 0 )
block3D = (block_size_x, block_size_y, block_size_z)
grid3D = (gridx, gridy, gridz)

#initialize pyCUDA context
cudaDevice = setCudaDevice( devN=useDevice, usingAnimation=True )
initialMemory = getFreeMemory( show=True )

#Read and compile CUDA code
print "\nCompiling CUDA code"
########################################################################
from pycuda.elementwise import ElementwiseKernel
floatToUchar = ElementwiseKernel(arguments="float *input, unsigned char *output",
				operation = "output[i] = (unsigned char) ( -255*(input[i]-1));",
				name = "floatToUchar_kernel")
########################################################################

########################################################################
def sendToScreen( ):
  # floatToUchar( plotDataFloat_d, plotData_d )
  copyToScreenArray()
  if volumeRender.nTextures==2:
		# floatToUchar( plotDataFloat_1, plotData_d_1 )
		copyToScreenArray_1()

def stepFunction():
  global  nSnap
  sendToScreen( )


#Initialize all gpu data
print "\nInitializing Data"
initialMemory = getFreeMemory( show=True )
# plotDataFloat_d = gpuarray.to_gpu(plotData_h)
# plotData_d = gpuarray.to_gpu(np.zeros([nDepth, nHeight, nWidth], dtype = np.uint8))
# volumeRender.plotData_dArray, copyToScreenArray = gpuArray3DtocudaArray( plotData_d )
volumeRender.plotData_dArray, copyToScreenArray = np3DtoCudaArray( plotData_h_256 )
if volumeRender.nTextures == 2:
  # plotDataFloat_1 = gpuarray.to_gpu(plotData_1)
  # plotData_d_1 = gpuarray.to_gpu(np.zeros([nDepth, nHeight, nWidth], dtype = np.uint8))
  volumeRender.plotData_dArray_1, copyToScreenArray_1 = np3DtoCudaArray( plotData_h_256_1 )
  # volumeRender.plotData_dArray, copyToScreenArray = gpuArray3DtocudaArray( plotData_d )
finalMemory = getFreeMemory( show=False )
print " Total Global Memory Used: {0} Mbytes\n".format(float(initialMemory-finalMemory)/1e6)


def specialKeyboardFunc( key, x, y ):
  global nSnap
  if key== volumeRender.GLUT_KEY_RIGHT:
    nSnap += 1
    if nSnap == nSnapshots: nSnap = 0
    print " Snapshot: ", nSnap
    change_snapshot( nSnap )
  # if key== volumeRender.GLUT_KEY_LEFT:
    # if volumeRender.transp_center > -0.95: volumeRender.transp_center -= 0.01
  # volumeRender.set_transfer_function( )

#configure volumeRender functions
volumeRender.specialKeys = specialKeyboardFunc
volumeRender.stepFunc = stepFunction
# volumeRender.keyboard = keyboard
#run volumeRender animation
sendToScreen( )
volumeRender.animate()
