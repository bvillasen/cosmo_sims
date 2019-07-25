import numpy as np
import h5py as h5
import yt

dataDir = '/home/bruno/Desktop/hard_drive_1/data/'
# dataDir = '/raid/bruno/data/'
inDir = dataDir + 'cosmo_sims/enzo/256_cool_uv_100Mpc/h5_files/'
enzoDir = inDir
outDir = inDir


outFileName = 'outputs_enzo_cool_uv_256_100Mpc.txt'
# a_list = [ 1./21 ]
a_list = []

nSnap = 0

# for nSnap in range(180):
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


for nSnap in range(32):
  fileName = 'snapshot_{0:03}.h5'.format( nSnap )
  file = h5.File( inDir+fileName, 'r' )
  current_a = file.attrs['current_a']
  print( nSnap, current_a )
  a_list.append( current_a )

n_vals = len( a_list )
a_vals = np.array(a_list)


np.savetxt( outDir + outFileName, a_vals )
# 
# 
# outDir = '/home/bruno/Desktop/'
# outFileName = 'scale_outputs_zeldovich_40.txt'
# z_start = 20
# z_end = 0
# n_points = 41
# z_vals =  np.linspace( z_start, z_end, n_points )
# a_vals = 1 / ( z_vals + 1)
# # a_start = 1./ ( z_start + 1)
# # a_end = 1./ ( z_end + 1)
# # a_vals = np.linspace( a_start, a_end, n_points )
# np.savetxt(  outDir+outFileName, a_vals )
