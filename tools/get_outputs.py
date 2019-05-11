import numpy as np
import h5py as h5

# dataDir = '/home/bruno/Desktop/hdd_extrn_1/data/'
dataDir = '/raid/bruno/data/'
inDir = dataDir + 'cosmo_sims/enzo/256_cool_uv/h5_files/'
outDir = inDir



# a_list = []
# 
# for nSnap in range(31):
#   fileName = 'snapshot_{0:03}.h5'.format( nSnap )
#   file = h5.File( inDir+fileName, 'r' )
#   current_a = file.attrs['current_a']
#   print nSnap, current_a
#   a_list.append( current_a )
# 
# n_vals = len( a_list )
# a_vals = np.array(a_list)
# 
# fileName = 'outputs_enzo_cool_UV_30.txt'.format( n_vals)
# np.savetxt( outDir+fileName, a_vals )


outDir = '/home/bruno/Desktop/'
outFileName = 'scale_outputs_200.txt'
z_start = 100
z_end = 0
a_start = 1./ ( z_start + 1)
a_end = 1./ ( z_end + 1)
n_points = 200
a_vals = np.linspace( a_start, a_end, n_points )
np.savetxt(  outDir+outFileName, a_vals )
