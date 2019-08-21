import os, sys
from os import listdir
from os.path import isfile, join
import h5py
import numpy as np


def compress_grid( nSnap, nBoxes, name_base, out_base_name,inDir, outDir, float_precision=False, fields='all' ):
  inFileName = '{0}.{1}.{2}'.format(nSnap, name_base, 0)
  inFile = h5py.File( inDir + inFileName, 'r')
  head = inFile.attrs
  head_keys = head.keys()
  dims_all = head['dims']
  dims_local = head['dims_local']
  keys = inFile.keys()
  
  added_header = False
  
  fileName = out_base_name + '{0}.h5'.format( nSnap )
  fileSnap = h5py.File( outDir + fileName, 'w' )
  
  if inFile.attrs.get('Current_z'): print ' snap: {0}    current_z: {1}'.format( nSnap, inFile.attrs['Current_z'][0] )  
  else: print ' snap: {0}  '.format( nSnap )
  
  for key in keys:
    if fields != 'all':
      if key not in fields: continue
    if float_precision: data_all = np.zeros( dims_all, dtype=np.float32 )
    else: data_all = np.zeros( dims_all, dtype=np.float64 )
    for nBox in range(nBoxes):
      inFileName = '{0}.{1}.{2}'.format(nSnap, name_base, nBox)
      inFile = h5py.File( inDir + inFileName, 'r')
      head = inFile.attrs
      if added_header == False:
        for h_key in head_keys:
          if h_key in ['dims', 'dims_local', 'offset', 'bounds', 'domain', 'dx', ]: continue
          # if inFile.attrs.get(h_key): 
          # print h_key
          fileSnap.attrs[h_key] = inFile.attrs[h_key][0]
        added_header = True
      procStart_z, procStart_y, procStart_x = head['offset']
      procEnd_z, procEnd_y, procEnd_x = head['offset'] + head['dims_local']
      data_local = inFile[key][...]
      data_all[ procStart_z:procEnd_z, procStart_y:procEnd_y, procStart_x:procEnd_x] = data_local
      inFile.close()
    fileSnap.create_dataset( key, data=data_all )
    fileSnap.attrs['max_'+ key ] = data_all.max()
    fileSnap.attrs['min_'+ key ] = data_all.min()
    fileSnap.attrs['mean_'+ key ] = data_all.mean()
  fileSnap.close()
  print ' Saved File: ', outDir+fileName, '\n'
