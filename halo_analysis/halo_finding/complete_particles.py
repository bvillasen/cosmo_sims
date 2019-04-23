import sys, os, time
import subprocess
import h5py as h5
import numpy as np



dataDir = '/home/bruno/Desktop/data/cosmo_sims/gadget/ay9_256/halos/snap_100/'
devDirectory = '/home/bruno/Desktop/Dropbox/Developer/'

nSnap = 100

x_list, y_list, z_list = [], [], []
particle_id_list = []
assigned_internal_haloid_list = []
internal_haloid_list = []
external_haloid_list = []

for box in range( 8 ):
  input_file = dataDir + 'halos_{0}.{1}.particles'.format( nSnap, box )  
  print "\nLoading File: ", input_file

  data = np.loadtxt( input_file ).T

  x, y, z, vx, vy, vz, particle_id, assigned_internal_haloid, internal_haloid, external_haloid = data 
  particle_id = particle_id.astype( np.int )
  assigned_internal_haloid = assigned_internal_haloid.astype( np.int )
  internal_haloid = internal_haloid.astype( np.int )
  external_haloid = external_haloid.astype( np.int )

  x_list.extend( list(x) )
  y_list.extend( list(y) )
  z_list.extend( list(z) )
  particle_id_list.extend( list(particle_id) )
  assigned_internal_haloid_list.extend( list(assigned_internal_haloid) )
  internal_haloid_list.extend( list(internal_haloid) )
  external_haloid_list.extend( list(external_haloid) )

x_arr = np.array( x_list )
y_arr = np.array( y_list )
z_arr = np.array( z_list )
particle_id_arr = np.array( particle_id_list )
assigned_internal_haloid_arr = np.array( assigned_internal_haloid_list )
internal_haloid_arr = np.array( internal_haloid_list )
external_haloid_arr = np.array( external_haloid_list )


output_file = dataDir + 'particles_{0}.h5'.format( nSnap )
file = h5.File( output_file, 'w' )

file.create_dataset( 'x', data=z_arr )
file.create_dataset( 'y', data=y_arr )
file.create_dataset( 'z', data=x_arr )
file.create_dataset( 'particle_id', data=particle_id_arr)
file.create_dataset( 'assigned_internal_haloid', data=assigned_internal_haloid_arr)
file.create_dataset( 'internal_haloid', data=internal_haloid_arr)
file.create_dataset( 'external_haloid', data=external_haloid_arr)

file.close()

