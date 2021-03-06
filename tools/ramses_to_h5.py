import os, sys
from os import listdir
from os.path import isfile, join
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import yt
from cosmo_constants import *

cosmo_dir = '/home/bruno/Desktop/Dropbox/Developer/cosmo_sims/'
toolsDirectory = cosmo_dir + "tools/"
sys.path.extend([toolsDirectory ] )

# dataDir = '/home/bruno/Desktop/data/'
# enzoDir = dataDir + 'cosmo_sims/enzo/256_hydro_step1/'
# inDir = enzoDir

# dataDir = '/raid/bruno/data/'
dataDir = '/home/bruno/Desktop/hard_drive_1/data/'
# inDir = dataDir + 'cosmo_sims/ramses/128_hydro_50Mpc_slope1_cfl25/'
inDir = dataDir + 'cosmo_sims/ramses/256_hydro_50Mpc_initialTemp/'
outDir = inDir + 'h5_files/'

# 
# dataFiles = [f for f in listdir(inDir) if  (f.find('DD') == 0 )   ]
# dataFiles = np.sort( dataFiles )
# nFiles = len( dataFiles )

# hydro = True
# cooling = True
# metals = True

hydro = True
cooling = False
metals = False

def get_internal_energy( temp, gamma=5./3 ):
  u = temp / (gamma - 1) * K_b / M_p
  return u



# n_snaps = 165
# snapshots = list(range(0,n_snaps, 2))
# if n_snaps-1 not in snapshots: snapshots.append(n_snaps-1)


snapshots = range(1,101)

z_list = []
t_list = []

nSnap = 2
for nSnap in snapshots:

  nSnap_out = nSnap -1

  snapKey = '{0:05}'.format(nSnap)
  inFileName = 'output_{0}/info_{0}.txt'.format( snapKey)

  print( "Loading file: ", inDir + inFileName )
  ds = yt.load( inDir + inFileName )
  data = ds.all_data()

  h = ds.hubble_constant
  current_z = np.float(ds.current_redshift)
  current_a = 1./(current_z + 1)
  gamma = 5./3

  # temp = data[('gas', 'temperature')].v
  # print temp.mean()
  # 
  if hydro:
    data_grid = ds.covering_grid( level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions )
    gas_dens = data_grid[ ('gas', 'density')].in_units('msun/kpc**3').v*current_a**3/h**2
    # gas_vel_x = data_grid[('gas','velocity_x')].in_units('km/s').v
    # gas_vel_y = data_grid[('gas','velocity_y')].in_units('km/s').v
    # gas_vel_z = data_grid[('gas','velocity_z')].in_units('km/s').v
    # gas_u = data_grid[('gas', 'thermal_energy' )].v * 1e-10 * gas_dens #km^2/s^2
    # gas_E = data_grid[('gas', 'total_energy' )].v * 1e-10   *gas_dens  #km^2/s^2
    # mu = data[('gas', 'mean_molecular_weight' )].v
    gas_temp = data_grid[ ('gas', 'temperature')].v
    print ' ramses: ', current_z
    print gas_temp.mean()
    t = ( gas_dens * gas_temp ).sum() / gas_dens.sum()
    z_list.append( current_z )
    t_list.append( t )  
  #   gas_u = get_internal_energy( gas_temp ) * gas_dens * 1e-6
  #   E_kin = 0.5*gas_dens*( gas_vel_x*gas_vel_x + gas_vel_y*gas_vel_y + gas_vel_z*gas_vel_z )
  #   gas_E = E_kin + gas_u
  #   print gas_dens.mean(),  gas_dens.min(), gas_dens.max()
  # 
  # if cooling :
  #   H_dens =  data_grid[ ('gas', 'H_density')].in_units('msun/kpc**3')*current_a**3/h**2
  #   H_0_dens =  data_grid[ ('gas', 'H_p0_density')].in_units('msun/kpc**3')*current_a**3/h**2
  #   H_1_dens =  data_grid[ ('gas', 'H_p1_density')].in_units('msun/kpc**3')*current_a**3/h**2
  #   He_dens =  data_grid[ ('gas', 'He_density')].in_units('msun/kpc**3')*current_a**3/h**2
  #   He_0_dens =  data_grid[ ('gas', 'He_p0_density')].in_units('msun/kpc**3')*current_a**3/h**2
  #   He_1_dens =  data_grid[ ('gas', 'He_p1_density')].in_units('msun/kpc**3')*current_a**3/h**2
  #   He_2_dens =  data_grid[ ('gas', 'He_p2_density')].in_units('msun/kpc**3')*current_a**3/h**2
  #   electron_dens =  data_grid[ ('gas', 'El_density')].in_units('msun/kpc**3')*current_a**3/h**2
  # 
  # if metals:
  #   metal_dens = data_grid[ ('gas', 'metal_density')].in_units('msun/kpc**3')*current_a**3/h**2
  # 
  # p_mass = data[('all', 'particle_mass')].in_units('msun')*h
  # p_pos_x = data[('all', 'particle_position_x')].in_units('kpc')/current_a*h
  # p_pos_y = data[('all', 'particle_position_y')].in_units('kpc')/current_a*h
  # p_pos_z = data[('all', 'particle_position_z')].in_units('kpc')/current_a*h
  # p_vel_x = data[('all', 'particle_velocity_x')].in_units('km/s')
  # p_vel_y = data[('all', 'particle_velocity_y')].in_units('km/s')
  # p_vel_z = data[('all', 'particle_velocity_z')].in_units('km/s')
  # 
  # 
  snapKey = '_{0:03}'.format( nSnap_out)
  base_name = 'snapshot'
  fileName = outDir + base_name + snapKey + '.h5'
  
  print (' Writing file: {0}'.format( fileName ))
  h5_file = h5.File( fileName, 'w')
  print( '  nSnap: {0}     current_a: {1}'.format(nSnap, current_a ))
  h5_file.attrs['current_a'] = current_a
  h5_file.attrs['current_z'] = current_z
  # 
  # dm = h5_file.create_group( 'dm' )
  # dm.create_dataset( 'mass', data=p_mass)
  # dm.create_dataset( 'pos_x', data=p_pos_x)
  # dm.create_dataset( 'pos_y', data=p_pos_y)
  # dm.create_dataset( 'pos_z', data=p_pos_z)
  # dm.create_dataset( 'vel_x', data=p_vel_x)
  # dm.create_dataset( 'vel_y', data=p_vel_y)
  # dm.create_dataset( 'vel_z', data=p_vel_z)
  
  if hydro:
    gas = h5_file.create_group( 'gas' )
    gas.attrs['gamma'] = gamma
    gas.create_dataset( 'density', data=gas_dens )
    # gas.create_dataset( 'momentum_x', data=gas_vel_x * gas_dens )
    # gas.create_dataset( 'momentum_y', data=gas_vel_y * gas_dens )
    # gas.create_dataset( 'momentum_z', data=gas_vel_z * gas_dens )
    # gas.create_dataset( 'Energy', data=gas_E )
    # gas.create_dataset( 'GasEnergy', data=gas_u  )
    gas.create_dataset( 'temperature', data=gas_temp  )
  
  # 
  # if cooling:
  #   gas.create_dataset( 'H_dens', data=H_0_dens )
  #   gas.create_dataset( 'HI_dens', data=H_1_dens )
  #   gas.create_dataset( 'He_dens', data=He_0_dens )
  #   gas.create_dataset( 'HeI_dens', data=He_1_dens )
  #   gas.create_dataset( 'HeII_dens', data=He_2_dens )
  #   gas.create_dataset( 'electron_dens', data=electron_dens )
  # 
  # if metals:
  #   gas.create_dataset( 'metal_dens', data=metal_dens )
  # 
  # 
  h5_file.close()
data = np.array([ z_list, t_list ])
out_file_name = 'avrg_temp_mass_ramses_100.dat'
np.savetxt(outDir + out_file_name, data)  