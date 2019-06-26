import os, sys
from os import listdir
from os.path import isfile, join
import h5py as h5
import numpy as np

currentDirectory = os.getcwd()
#Add Modules from other directories
cosmoDir = currentDirectory[: currentDirectory.find('ics')]
toolsDirectory = cosmoDir + 'tools/'
sys.path.append( toolsDirectory )
from tools import create_directory
from domain_decomposition import get_domain_block, get_domain_parent


dataDir = '/home/bruno/Desktop/hard_drive_1/data/'
inDir = dataDir + 'summit/ics/128_25Mpc/ics/'
outDir = dataDir + 'summit/ics/384_75Mpc/ics/'
create_directory( outDir )

out_grid_size = np.array([ 3, 3, 3 ])
nz_g, ny_g, nx_g = out_grid_size
box_size = 25000 #25Mpc


nSnap = 0


part_file_name = inDir + '{0}_particles.h5'.format(nSnap)
part_file = h5.File( part_file_name, 'r' )
header = part_file.attrs

part_out_file_name = outDir + '{0}_particles.h5'.format(nSnap)
part_out_file =  h5.File( part_out_file_name, 'w' )
for key in header.keys():
  if key == 'n_particles_local': continue
  part_out_file.attrs[key] = header[key]
  # print key, header[key]

for field in part_file.keys():
  data_out = []
  for k in range(nz_g):
      for j in range(ny_g):
        for i in range(nx_g):
          data = part_file[field][...]
          if field == 'pos_x': data += i*box_size
          if field == 'pos_y': data += j*box_size
          if field == 'pos_z': data += k*box_size 
          data_out.append( data )
  data_out = np.array( data_out ).flatten()
  n_particles_local = len(data_out)
  print field, data_out.shape, data_out.min(), data_out.max()  
  part_out_file.create_dataset( field , data=data_out.astype(np.float64) )
part_out_file.attrs['n_particles_local'] = n_particles_local
print 'N particles: ', n_particles_local  
part_out_file.close()
print "Saved File: ", part_out_file_name

# grid_file_name = inDir + '{0}.h5'.format(nSnap)
# grid_file = h5.File( grid_file_name, 'r' )
# header = grid_file.attrs
# 
# grid_out_file_name = outDir + '{0}.h5'.format(nSnap)
# grid_out_file =  h5.File( grid_out_file_name, 'w' )
# for key in header.keys():
#   grid_out_file.attrs[key] = header[key]
# 
# for field in grid_file.keys():
#   data = grid_file[field][...]
#   data_size = np.array( data.shape )
#   nz, ny, nx = data_size
#   out_data_size = data_size * out_grid_size
#   data_out = np.zeros( out_data_size )
#   for k in range(nz_g):
#     for j in range(ny_g):
#       for i in range(nx_g):
#         data_out[k*nz:(k+1)*nz, j*ny:(j+1)*ny, i*nx:(i+1)*nx ] = data
#   grid_out_file.create_dataset( field , data=data_out.astype(np.float64) )
# 
# grid_out_file.close()
# print "Saved File: ", grid_out_file_name