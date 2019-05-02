import os, sys
from os import listdir
from os.path import isfile, join
import h5py
import numpy as np

def compress_particles(  nSnap, nBoxes, name_base, out_base_name,inDir, outDir, cosmology=True):


  inFileName = '{0}_particles.{1}.{2}'.format(nSnap, name_base, 0)
  inFile = h5py.File( inDir + inFileName, 'r')
  head = inFile.attrs
  dims_all = head['dims']
  dims_local = head['dims_local']

  nz, ny, nx = dims_all
  inFile.close()

  
  fileName = out_base_name + '{0}.h5'.format( nSnap )
  fileSnap = h5py.File( outDir + fileName, 'w')
  keys = [  'density' ]
  # keys_parts = [ 'pos_x', 'pos_y', 'pos_z',  'vel_x', 'vel_y', 'vel_z' ]
  keys_parts = [  ]
  keys_all = keys + keys_parts
  added_time = False
  print ' snap: {0}  {1}'.format( nSnap, keys_all )
  for key in keys_all:
    data_all = np.zeros( dims_all )
    data_all_parts = []
    for nBox in range(nBoxes):
      inFileName = '{0}_particles.{1}.{2}'.format(nSnap, name_base, nBox)
      inFile = h5py.File( inDir + inFileName, 'r')
      head = inFile.attrs
      if cosmology:
        current_a = head['current_a']
        current_z = head['current_z']
      # particle_mass= head['particle_mass']
      procStart_z, procStart_y, procStart_x = head['offset']
      procEnd_z, procEnd_y, procEnd_x = head['offset'] + head['dims_local']
      if key in keys:
        data_local = inFile[key][...]
        data_all[ procStart_z:procEnd_z, procStart_y:procEnd_y, procStart_x:procEnd_x] = data_local
      if key in keys_parts:
        data_local_parts = inFile[key][...]
        data_all_parts.append(data_local_parts)
      inFile.close()
    if key in keys:
      fileSnap.create_dataset( key, data=data_all )
      fileSnap.attrs['max_'+ key ] = data_all.max()
      fileSnap.attrs['min_'+ key ] = data_all.min()
      fileSnap.attrs['mean_'+ key ] = data_all.mean()
    if key in keys_parts:
      array_parts = np.concatenate(data_all_parts)
      print 'nParticles: ', len(array_parts)
      fileSnap.create_dataset( key, data=array_parts )
    if added_time == False:
      if cosmology:
        fileSnap.attrs['current_z'] = current_z[0]
        fileSnap.attrs['current_a'] = current_a[0]
      # fileSnap.attrs['particle_mass'] = particle_mass[0]
      added_time = True
  fileSnap.close()
  print ' Saved File: ', outDir+fileName
