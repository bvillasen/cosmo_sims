import sys, time, os
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
import volumeRender
from cudaTools import setCudaDevice, getFreeMemory, gpuArray3DtocudaArray, np3DtoCudaArray
# from load_data_cholla import load_data_snap

nFields = 2
transp_type = 'sigmoid'
useDevice = 0
for option in sys.argv:
  if option == "anim": usingAnimation = True
  if option.find("dev=") != -1: useDevice = int(option[-1])
  if option.find("n=") != -1: nFields = int(option[-1])
  if option.find("sigmoid") != -1: transp_type = 'sigmoid'
  if option.find("iso") != -1: transp_type = 'gaussian'

#
# inDir = dataDir + 'cosmo_sims/cholla_pm/cosmo_512_hydro/'
# outDir = '/home/bruno/Desktop/anim_cosmo/'

#
# gridFileName = inDir + 'data_grid.h5'
# partFileName = inDir + 'data_particles.h5'
#
# # gridFileName = inDir + 'data_grid_DE_sync1_pot2.h5'
# # partFileName = inDir + 'data_particles_DE_sync1_pot2.h5'
#
# # nSnap = 101
# # snapKey = str(nSnap)
# #
# # data_all, nSnapshots, current_z = load_data_snap( snapKey, gridFileName, partFileName )
# # # nSnapshots = 3
# # data_grid = data_all['grid']
# # current_a = 1/( current_z +1 )
# # print current_z
# #
# # dens_gas = data_grid['density'][...]
# # temp = data_grid['GasEnergy'][...]
# # # pot = data_grid['potential'][...]
# # # momentum_x = data_grid['momentum_x'][...]
# # # momentum_y = data_grid['momentum_y'][...]
# # # momentum_z = data_grid['momentum_z'][...]
# # # energy = data_grid['Energy']
# # data_dm = data_all['dm']
# # dens_dm = data_dm['density'][...]
# # dens_grav = data_grid['grav_density'][...]
#
#
# inFileName = inDir + 'data_log10.h5'
# inFile = h5.File( inFileName, 'r' )
# max_gas_dens_vals = inFile['gas']['max_density']
# max_gas_temp_vals = inFile['gas']['max_GasEnergy']
# max_dm_dens_vals = inFile['dm']['max_density']
#
# def prepare_data( nSnap, plotData, max_vals, log=False ):
#   # plotData = np.abs(plotData.astype(np.float32))
#   if log : plotData = np.log10(plotData + 1)
#   # plotData -= plotData.min()
#   max_val = max_vals[nSnap]
#   norm_val = max_vals[nSnap:nSnap+5].mean()
#   norm_val = max_vals[-1]
#   plotData /= norm_val
#   # plotData *= 0.9
#   # plotData += 0.0
#   plotData_h_256 = (-255*(plotData-1)).astype(np.uint8)
#   # plotData_h = prepare_data( plotData_h )
#   # plotData_1 = plotData_h.copy()
#
#   n=3
#   plotData_h_256[:,:n,:n] = 0
#   plotData_h_256[:n,:,:n] = 0
#   plotData_h_256[:n,:n,:] = 0
#   plotData_h_256[-n:,-n:,:] = 0
#   plotData_h_256[:,-n:,-n:] = 0
#   plotData_h_256[-n:,:,-n:] = 0
#   plotData_h_256[-n:,:n,:] = 0
#   plotData_h_256[-n:,:,:n] = 0
#   plotData_h_256[:,:n,-n:] = 0
#   plotData_h_256[:,-n:,:n] = 0
#   plotData_h_256[:n,:,-n:] = 0
#   plotData_h_256[:n,-n:,:] = 0
#
#   return plotData_h_256
#
# def get_data( inFile, nSnap, key_1, key_2 ):
#   snapKey = str(nSnap)
#   snapData = inFile[snapKey]
#   if key_1 == 'dm_dens':
#     data_1 = snapData['dm']['density'][...]
#     max_vals_1 = inFile['dm']['max_density'][...]
#   if key_1 == 'gas_dens':
#     data_1 = snapData['gas']['density'][...]
#     max_vals_1 = inFile['gas']['max_density'][...]
#   if key_2 == 'gas_dens':
#     data_2 = snapData['gas']['density'][...]
#     max_vals_2 = inFile['gas']['max_density'][...]
#   if key_2 == 'gas_temp':
#     data_2 = snapData['gas']['GasEnergy'][...]
#     max_vals_2 = inFile['gas']['max_GasEnergy'][...]
#   plotData_256 = prepare_data( nSnap, data_1, max_vals_1, log=False ).copy()
#   plotData_256_1 = prepare_data( nSnap, data_2, max_vals_2, log=False ).copy()
#   return plotData_256, plotData_256_1
#
#
# # def get_data_pair( inFile, nSnap, key, time ):
# #   snapData_0 inFile[str(nSnap)]
# #   snapData_1 inFile[str(nSnap+1)]
# #   if key == 'dm_dens':
# #     data_0 = snapData_0['dm']['density'][...]
# #     data_1 = snapData_1['dm']['density'][...]
# #   if key == 'gas_dens':
# #     data_0 = snapData_0['gas']['density'][...]
# #     data_1 = snapData_1['gas']['density'][...]
# #   if key == 'gas_temp':
# #     data_0 = snapData_0['gas']['GasEnergy'][...]
# #     data_1 = snapData_1['gas']['GasEnergy'][...]
# #   return time*data_0 + (data_1 - data_0)*(1-time)
#
#
#
#
#
#
# # data_1, data_2 = get_data( inFile, nSnap, 'dm_dens', 'gas_dens' )
# #
# # plotData_h = data_1
# # plotData_h_1 = data_2
#
# # #
# # plotData_h_256 = prepare_data( plotData_h, log=False ).copy()
# # plotData_h_256_1 = prepare_data( plotData_h_1, log=False ).copy()
#
# nSnap = 0
# plotData_h_256, plotData_h_256_1 = get_data( inFile, nSnap, 'dm_dens', 'gas_dens' )
#
# nz, ny, nx = plotData_h_256.shape
# nWidth, nHeight, nDepth = nx, ny, nz
#
# def change_snap( nSnap):
#   global plotData_h_256, plotData_h_256_1, copyToScreenArray, copyToScreenArray_1
#   # snapKey = str(nSnap)
#   # data_all, nSnapshots, current_z = load_data_snap( snapKey, gridFileName, partFileName )
#   # data_grid = data_all['grid']
#   # data_dm = data_all['dm']
#   # dens_dm = data_dm['density'][...]
#   # dens_gas = data_grid['density'][...]
#   # dens_grav = data_grid['grav_density'][...]
#   # print dens_dm.mean(), dens_grav.mean(), dens_gas.mean()
#   # plotData_h = dens_dm
#   # plotData_h_1 = dens_gas
#   # plotData_h_256 = prepare_data( plotData_h, log=True )
#   # plotData_h_256_1 = prepare_data( plotData_h_1, log=True )
#   plotData_h_256, plotData_h_256_1 = get_data( inFile, nSnap, 'dm_dens', 'gas_dens' )
#   copyToScreenArray.set_src_host(plotData_h_256)
#   copyToScreenArray_1.set_src_host(plotData_h_256_1)
#   copyToScreenArray()
#   copyToScreenArray_1()
#
#
nx = ny = nz = 128
nWidth, nHeight, nDepth = nx, ny, nz

data_float_1 = 256 * np.random.rand( nx, ny, nz)
data_float_2 = 256 * np.random.rand( nx, ny, nz)
data_int_1_h = data_float_1.astype(np.uint8)
data_int_2_h = data_float_2.astype(np.uint8)
#
# data_int_1_d = gpuarray.to_gpu( data_int_1_h )
# data_int_2_d = gpuarray.to_gpu( data_int_2_h)
# volumeRender.plotData_dArray, copyToScreenArray = gpuArray3DtocudaArray( data_int_1_d )

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
#
# #set thread grid for CUDA kernels
# block_size_x, block_size_y, block_size_z = 8,8,8   #hardcoded, tune to your needs
# gridx = nWidth // block_size_x + 1 * ( nWidth % block_size_x != 0 )
# gridy = nHeight // block_size_y + 1 * ( nHeight % block_size_y != 0 )
# gridz = nDepth // block_size_z + 1 * ( nDepth % block_size_z != 0 )
# block3D = (block_size_x, block_size_y, block_size_z)
# grid3D = (gridx, gridy, gridz)
#
#
# #initialize pyCUDA context
# cudaDevice = setCudaDevice( devN=useDevice, usingAnimation=True )
# initialMemory = getFreeMemory( show=True )
#
# #Read and compile CUDA code
# print "\nCompiling CUDA code"
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
# ########################################################################
# def sendToScreen( ):
#   # floatToUchar( plotDataFloat_d, plotData_d )
#   copyToScreenArray()
#   if volumeRender.nTextures==2:
# 		# floatToUchar( plotDataFloat_1, plotData_d_1 )
# 		copyToScreenArray_1()
# ########################################################################
# nSnaps = 360*10
# delta_theta = 360./nSnaps
# nData = 100
# nChanges = nSnaps/nData
# counter = 0
# def stepFunction():
#   global counter, nSnap
#   sendToScreen( )
#   volumeRender.viewRotation[1] += delta_theta
#   # time.sleep(0.01)
#   counter += 1
#   if counter%nChanges == 0 :
#     nSnap += 1
#     change_snap(nSnap)
#   if counter > 0 :volumeRender.save_image(dir=outDir)
#
# #Initialize all gpu data
# print "\nInitializing Data"
# initialMemory = getFreeMemory( show=True )
# # plotDataFloat_d = gpuarray.to_gpu(plotData_h)
# # plotData_d = gpuarray.to_gpu(np.zeros([nDepth, nHeight, nWidth], dtype = np.uint8))
# volumeRender.plotData_dArray, copyToScreenArray = np3DtoCudaArray( plotData_h_256 )
# # volumeRender.plotData_dArray, copyToScreenArray = gpuArray3DtocudaArray( plotData_d )
# if volumeRender.nTextures == 2:
#   # plotDataFloat_1 = gpuarray.to_gpu(plotData_1)
#   # plotData_d_1 = gpuarray.to_gpu(np.zeros([nDepth, nHeight, nWidth], dtype = np.uint8))
#   volumeRender.plotData_dArray_1, copyToScreenArray_1 = np3DtoCudaArray( plotData_h_256_1 )
# finalMemory = getFreeMemory( show=False )
# print " Total Global Memory Used: {0} Mbytes\n".format(float(initialMemory-finalMemory)/1e6)
#
#
# def specialKeyboardFunc( key, x, y ):
#   global plotData_h, nSnap
#   if key== volumeRender.GLUT_KEY_UP:
#     volumeRender.transp_ramp += 0.05
#   if key== volumeRender.GLUT_KEY_DOWN:
#   	if volumeRender.transp_ramp > 0.055: volumeRender.transp_ramp -= 0.05
#   if key== volumeRender.GLUT_KEY_RIGHT:
#   	if volumeRender.transp_center < 0.95: volumeRender.transp_center += 0.01
#   if key== volumeRender.GLUT_KEY_LEFT:
#     if volumeRender.transp_center > -0.95: volumeRender.transp_center -= 0.01
#   volumeRender.set_transfer_function( )
#
#
# # def keyboard(*args):
# #   # global transferScale, brightness, density, transferOffset, cmap_indx
# #   global nSnap, cmap_indx
# #   ESCAPE = '\033'
# #   # If escape is pressed, kill everything.
# #   if args[0] == ESCAPE:
# #     print "Ending Simulation"
# #     #cuda.gl.Context.pop()
# #     sys.exit()
# #   if args[0] == '+':
# #     nSnap += 1
# #     if nSnap >= nSnapshots: nSnap = 0
# #     print 'Snapshot: ', nSnap
# #     change_snap()
# #   if args[0] == 's': volumeRender.save_image()
#
#
# #configure volumeRender functions
# volumeRender.specialKeys = specialKeyboardFunc
# volumeRender.stepFunc = stepFunction
# # volumeRender.keyboard = keyboard
# #run volumeRender animation
# sendToScreen( )
# volumeRender.animate()
