import numpy as np
import h5py as h5
from tools import *
from load_data_cholla import load_snapshot_data_particles

dataDir = '/home/bruno/Desktop/hard_drive_1/data/'
# dataDir = '/raid/bruno/data/'
inDir = dataDir + 'cosmo_sims/cholla_pm/256_dm_50Mpc/data/'
enzoDir = inDir
outDir = inDir


fileKey = "particles"

outFileName = 'stats_{0}.h5'.format(fileKey)
dataFiles, nFiles = get_files_names( fileKey, inDir, type='cholla' )

print ('N Files: ', nFiles)

fields = ['density']

stats = {}

for nSnap in range(nFiles):
  print nSnap
  data_cholla = load_snapshot_data_particles( nSnap, inDir )
  min_vals, max_vals = [], [] 
  for field in fields:
    data = data_cholla[field][...]
    min_vals.append( data.min() )
    max_vals.append( data.max() )
  min_vals = np.array( min_vals )
  max_vals = np.array( max_vals )
  stats[field] = {}
  stats[field]['min_vals'] = min_vals
  stats[field]['max_vals'] = max_vals
  stats[field]['min_global'] = min_vals.min()
  stats[field]['max_global'] = max_vals.max()
  
  
  
outFile = h5.File( outDir + outFileName, 'w' )
for field in fields:
  group = outFile.create_group( field )
  group.attrs['min_global'] = stats[field]['min_global']
  group.attrs['max_global'] = stats[field]['max_global']
  group.create_dataset( 'min_vals', data =stats[field]['min_vals'] )
  group.create_dataset( 'max_vals', data =stats[field]['max_vals'] )


outFile.close()
  
  
  
  
  