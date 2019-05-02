import os, sys
from os import listdir
from os.path import isfile, join
import h5py
import numpy as np


def compress_grid( nSnap, nBoxes, name_base, out_base_name,inDir, outDir ):
  inFileName = '{0}.{1}.{2}'.format(nSnap, name_base, 0)
  inFile = h5py.File( inDir + inFileName, 'r')
  head = inFile.attrs
  dims_all = head['dims']
  dims_local = head['dims_local']
  keys = inFile.keys()
  

  added_time = False

  fileName = out_base_name + '{0}.h5'.format( nSnap )
  fileSnap = h5py.File( outDir + fileName, 'w' )
  print ' snap: {0}  '.format( nSnap )
  for key in keys:
    # if key in [ 'momentum_x', 'momentum_y', 'momentum_z', 'Energy', 'grav_potential', 'e_density', 'metal_density', 'HeI_density', 'HeII_density', 'HeIII_density']: continue
    if key in [  'grav_potential', ]: continue
    # print '  {0}'.format(key)
    # data_all = np.zeros( dims_all, dtype=np.float32 )
    # if key in [ 'flags_DE' ]: data_all = np.zeros( dims_all, dtype=np.bool )
    data_all = np.zeros( dims_all, dtype=np.float64 )
    for nBox in range(nBoxes):
      inFileName = '{0}.{1}.{2}'.format(nSnap, name_base, nBox)
      inFile = h5py.File( inDir + inFileName, 'r')
      head = inFile.attrs
      time = head['t']
      dt = head['dt']
      if added_time == False:
        fileSnap.attrs['t'] = time
        fileSnap.attrs['dt'] = dt
        for key_cosmo in [ 'H0', 'Omega_M', 'Omega_L', 'Current_a', 'Current_z']:
          if inFile.attrs.get(key_cosmo): 
            fileSnap.attrs[key_cosmo] = inFile.attrs[key_cosmo]
        added_time = True
      procStart_z, procStart_y, procStart_x = head['offset']
      procEnd_z, procEnd_y, procEnd_x = head['offset'] + head['dims_local']
      data_local = inFile[key][...]
      data_all[ procStart_z:procEnd_z, procStart_y:procEnd_y, procStart_x:procEnd_x] = data_local
      inFile.close()
    if key=='grav_density': print '  {0}   {1}   {2}'.format(data_all.mean(), data_all.min(), data_all.max())
    # if key=='flags_DE': print ' Flags_DE:  {0} '.format(data_all.sum())
    fileSnap.create_dataset( key, data=data_all )
    fileSnap.attrs['max_'+ key ] = data_all.max()
    fileSnap.attrs['min_'+ key ] = data_all.min()
    fileSnap.attrs['mean_'+ key ] = data_all.mean()
  fileSnap.close()
  print ' Saved File: ', outDir+fileName, '\n'
