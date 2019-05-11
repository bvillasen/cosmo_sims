import os, sys
from os import listdir
from os.path import isfile, join
import h5py
import numpy as np
import subprocess
import yt
from itertools import product

currentDirectory = os.getcwd()
#Add Modules from other directories
cosmoDir = currentDirectory[: currentDirectory.find('ics')]
toolsDirectory = cosmoDir + 'tools/'
sys.path.append( toolsDirectory )
# from load_data_enzo_old import load_snapshot_enzo, load_snapshot_enzo_yt
from expand_data_grid import expand_data_grid_to_cholla
from expand_data_particles import expand_data_particles_to_cholla
from generate_ics_particles_functions import generate_ics_particles, generate_ics_particles_single_domain
from expand_data_grid import expand_data_grid_to_cholla
from domain_decomposition import get_domain_block, get_domain_parent
from tools import create_directory

# dataDir = '/home/bruno/Desktop/data/'
dataDir = '/raid/bruno/data/'
enzoDir = dataDir + 'cosmo_sims/enzo/ZeldovichPancake/'
inDir = enzoDir
outputDir = dataDir + 'cosmo_sims/cholla_pm/zeldovich/ics/'
create_directory( outputDir )



def convert_3D( x, nPoints):
  x_2D = np.array( [x,]*nPoints)
  x_3D = np.array( [x_2D,]*nPoints)
  return x_3D


nSnap_enzo = 0

nPoints = 256

nSnap = nSnap_enzo

snapKey = '{0:03}'.format(nSnap)
inFileName = 'DD0{0}/data0{0}'.format( snapKey)

ds = yt.load( inDir + inFileName )
data = ds.all_data()
h = ds.hubble_constant
current_z = np.float(ds.current_redshift)
current_a = 1./(current_z + 1)


x = data[('gas', 'x')].in_units('Mpc/h').v / current_a
dens = data[ ('gas', 'density')].in_units('msun/kpc**3').v*current_a**3/h**2
vel_x = data[ ('gas', 'velocity_x')].in_units('km/s').v
vel_y = np.zeros_like( vel_x )
vel_z = np.zeros_like( vel_x )
U = data[('gas', 'thermal_energy' )].v * 1e-10 * dens #km^2/s^2 
E = data[('gas', 'total_energy' )].v * 1e-10 * dens #km^2/s^2 

data = np.array([ dens, vel_x, E, U]).flatten()

np.savetxt( 'ics_zeldovich.dat', data )


# 
# 
# gas_dens = convert_3D( dens, nPoints )
# gas_vel_x = convert_3D( vel_x, nPoints )
# gas_vel_y = convert_3D( vel_y, nPoints )
# gas_vel_z = convert_3D( vel_z, nPoints )
# gas_E = convert_3D( E, nPoints )
# gas_U = convert_3D( U, nPoints )
# p_pos = np.array([ pos for pos in product( x, x , x) ]).T
# p_pos_x = p_pos[0] * 1e3 #kpc/h
# p_pos_y = p_pos[1] * 1e3 #kpc/h
# p_pos_z = p_pos[2] * 1e3 #kpc/h
# p_vel_x = np.zeros_like( p_pos_x )
# p_vel_y = np.zeros_like( p_pos_y )
# p_vel_z = np.zeros_like( p_pos_z )
# p_mass = np.zeros_like( p_pos_x )
# 
# data_enzo = { 'dm':{}, 'gas':{} }
# data_enzo['current_a'] = current_a
# data_enzo['current_z'] = current_z
# 
# data_enzo['dm']['mass'] = p_mass
# data_enzo['dm']['pos_x'] = p_pos_x
# data_enzo['dm']['pos_y'] = p_pos_y
# data_enzo['dm']['pos_z'] = p_pos_z
# data_enzo['dm']['vel_x'] = p_vel_x
# data_enzo['dm']['vel_y'] = p_vel_y
# data_enzo['dm']['vel_z'] = p_vel_z
# 
# data_enzo['gas']['density'] = gas_dens
# data_enzo['gas']['momentum_x'] = gas_dens * gas_vel_x
# data_enzo['gas']['momentum_y'] = gas_dens * gas_vel_y
# data_enzo['gas']['momentum_z'] = gas_dens * gas_vel_z
# data_enzo['gas']['GasEnergy'] = gas_U
# data_enzo['gas']['Energy'] = gas_E
# 
# 
# proc_grid = [ 2, 2, 2]
# box_size = [ 64000, 64000, 64000 ]
# grid_size = [ 256, 256, 256 ]
# outputBaseName = '{0}_particles.h5'.format(nSnap)
# generate_ics_particles(data_enzo, outputDir, outputBaseName, proc_grid, box_size, grid_size)
# 
# outputBaseName = '{0}.h5'.format(nSnap)
# expand_data_grid_to_cholla( proc_grid, data_enzo['gas'], outputDir, outputBaseName )
