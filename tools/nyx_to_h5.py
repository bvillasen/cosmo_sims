import os, sys
from os import listdir
from os.path import isfile, join
import h5py
import numpy as np
currentDirectory = os.getcwd()
#Add Modules from other directories
toolsDirectory = currentDirectory
sys.path.append( toolsDirectory )
from tools import *
# from directories import cosmoDir, dataDir
from load_data_nyx import load_data_nyx_yt


dataDir = '/raid/bruno/data/'

base_name = 'snapshot_'

type = 'dm'
# type = 'hydro'
inDir = dataDir + 'cosmo_sims/nyx/256_{0}_50Mpc/'.format(type)
# inDir = dataDir + 'cosmo_sims/nyx/lya/'
outDir = inDir + 'h5_files/'

is_directory = True
fileKey = 'plt'


hydro = False
if type == 'hydro': hydro = True

dataFiles, nFiles = get_files_names( fileKey, inDir, type='nyx' )

print 'Saving Files: ', nFiles
current_a_list = []

for nSnap, inFileName in enumerate( dataFiles):
  # snapKey = '{0:05}'.format(nSnap*index_stride)
  # inFileName = 'plt' + snapKey
  print "\n Loading: ", inDir +inFileName
  data_nyx = load_data_nyx_yt( inFileName, inDir, hydro=hydro )
  current_a = data_nyx['current_a']
  current_z = data_nyx['current_z']
  snapKey = '{0:03}'.format(nSnap)
  current_a_list.append( current_a )
  print ' Current_a: ', current_a

  outputFileName = outDir + base_name + snapKey + '.h5'

  print '\nWriting h5 file: ', outputFileName
  outFile = h5py.File( outputFileName, 'w')
  outFile.attrs['current_a'] = current_a
  outFile.attrs['current_z'] = current_z
  #
  nPart_dm = len(data_nyx['dm']['mass'])
  print " N particles DM: ", nPart_dm
  data_dm = data_nyx['dm']
  mass_dm = data_dm['mass']
  pos_x_dm = data_dm['pos_x']
  pos_y_dm = data_dm['pos_y']
  pos_z_dm = data_dm['pos_z']
  vel_x_dm = data_dm['vel_x']
  vel_y_dm = data_dm['vel_y']
  vel_z_dm = data_dm['vel_z']
  print pos_x_dm.max(), pos_x_dm.min()

  dm = outFile.create_group( 'dm' )
  dm.create_dataset( 'mass', data=mass_dm )
  dm.create_dataset( 'pos_x', data=pos_x_dm )
  dm.create_dataset( 'pos_y', data=pos_y_dm )
  dm.create_dataset( 'pos_z', data=pos_z_dm )
  dm.create_dataset( 'vel_x', data=vel_x_dm )
  dm.create_dataset( 'vel_y', data=vel_y_dm )
  dm.create_dataset( 'vel_z', data=vel_z_dm )

  if hydro:
    data_gas = data_nyx['gas']
    dens_gas = data_gas['density']
    vel_x_gas = data_gas['vel_x']
    vel_y_gas = data_gas['vel_y']
    vel_z_gas = data_gas['vel_z']
    temp_gas = data_gas['temperature']
    gamma = 5./3
    mom_x_gas = dens_gas * vel_x_gas
    mom_y_gas = dens_gas * vel_y_gas
    mom_z_gas = dens_gas * vel_z_gas
    u_gas = get_internal_energy( temp_gas, gamma ) * 1e-6 * dens_gas
    E_gas = 0.5*(dens_gas) * ( vel_x_gas*vel_x_gas + vel_y_gas*vel_y_gas + vel_z_gas*vel_z_gas ) + u_gas

    # GasEnergy = E_gas - 0.5*dens_gas * ( vel_x_gas*vel_x_gas + vel_y_gas*vel_y_gas + vel_z_gas*vel_z_gas )
    # temp_cholla = get_temp(GasEnergy*1e6, 5./3)
    # temp_avrg = temp_cholla.mean()
    # print " Average Temp "

    gas = outFile.create_group( 'gas' )
    gas.create_dataset( 'density', data=dens_gas )
    # gas.create_dataset( 'pos_x', data=pos_x_gas )
    # gas.create_dataset( 'pos_y', data=pos_y_gas )
    # gas.create_dataset( 'pos_z', data=pos_z_gas )
    gas.create_dataset( 'momentum_x', data=mom_x_gas )
    gas.create_dataset( 'momentum_y', data=mom_y_gas )
    gas.create_dataset( 'momentum_z', data=mom_z_gas )
    gas.create_dataset( 'GasEnergy', data=u_gas )
    gas.create_dataset( 'Energy', data=E_gas )

  outFile.close()
  print 'Saved h5 file: ', outputFileName

# chollaDir = '/home/bruno/cholla/'
# scale_fle_name = chollaDir + 'scale_output_files/outputs_{0}_nyx.txt'.format(type)
# print "Saving scale_output_files: ", scale_fle_name
# np.savetxt( scale_fle_name, current_a_list )
