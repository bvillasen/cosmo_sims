import numpy as np
import h5py as h5
import yt

dataDir = '/home/bruno/Desktop/hard_drive_1/data/'
# dataDir = '/raid/bruno/data/'
inDir = dataDir + 'cosmo_sims/enzo/256_cool_uv_50Mpc/h5_files/'
enzoDir = inDir
outDir = inDir


outFileName = 'outputs_cool_uv_enzo_256_50Mpc.txt'
a_list = []

nSnapshots = 44
for nSnap in range(nSnapshots):
  fileName = 'snapshot_{0:03}.h5'.format( nSnap )
  file = h5.File( inDir+fileName, 'r' )
  current_a = file.attrs['current_a']
  print( nSnap, current_a )
  a_list.append( current_a )

n_vals = len( a_list )
a_vals = np.array(a_list)

np.savetxt( outDir + outFileName, a_vals )







# nSnap = 0
# 
# for nSnap in range(158):
#   # if nSnap == 0:
#   #   file_name = enzoDir + 'ics/DD0{0:03}/data0{0:03}'.format(nSnap)
#   # else:
#   #   nSnap -= 1  
#   #   file_name = enzoDir + 'DD0{0:03}/data0{0:03}'.format(nSnap)
# 
#   file_name = enzoDir + 'DD0{0:03}/data0{0:03}'.format(nSnap)
# 
#   ds = yt.load( file_name )
#   data = ds.all_data()
#   h = ds.hubble_constant
#   current_z = ds.current_redshift
#   current_a = 1./(current_z + 1)
#   a_list.append( current_a)
# 
# a_vals = np.array( a_list )
# np.savetxt(  outDir + outFileName, a_vals )

