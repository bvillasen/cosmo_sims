import os, sys
from os import listdir
from os.path import isfile, join
import h5py
import numpy as np
import subprocess
import yt

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
enzoDir = dataDir + 'cosmo_sims/enzo/128_cool_uv_50Mpc/ics/'
inDir = enzoDir
outputDir = dataDir + 'cosmo_sims/cholla_pm/128_cool_uv_50Mpc/ics_enzo/'
create_directory( outputDir )
nSnap_enzo = 0

cool = False
metals = False

nSnap = nSnap_enzo

snapKey = '{0:03}'.format(nSnap)
inFileName = 'DD0{0}/data0{0}'.format( snapKey)

ds = yt.load( inDir + inFileName )
data = ds.all_data()
h = ds.hubble_constant
current_z = np.float(ds.current_redshift)
current_a = 1./(current_z + 1)

data_grid = ds.covering_grid( level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions )
gas_dens = data_grid[ ('gas', 'density')].in_units('msun/kpc**3').v*current_a**3/h**2
gas_vel_x = data_grid[('gas','velocity_x')].in_units('km/s').v
gas_vel_y = data_grid[('gas','velocity_y')].in_units('km/s').v
gas_vel_z = data_grid[('gas','velocity_z')].in_units('km/s').v
# gas_E = data_grid[('gas', 'total_energy' )].v * 1e-10   *gas_dens  #km^2/s^2
gas_u = data_grid[('gas', 'thermal_energy' )].v * 1e-10 * gas_dens #km^2/s^2


vel_max = max( [ gas_vel_x.max(), gas_vel_y.max(), gas_vel_z.max() ])

print ' Vel max: ', vel_max

# gas_vel_x = np.ones_like( gas_dens ) * 1
# gas_vel_y = np.ones_like( gas_dens ) * 1
# gas_vel_z = np.ones_like( gas_dens ) * 1

gas_E = 0.5 * gas_dens * ( gas_vel_x*gas_vel_x + gas_vel_y*gas_vel_y + gas_vel_z*gas_vel_z ) + gas_u

# gas_temp = data_grid[ ('gas', 'temperature')].v
# temp_factor = 1.2385545089162293
# gas_temp *= temp_factor
# mu = data_grid[('gas', 'mean_molecular_weight' )].v

if cool:
  # H_dens =  data_grid[ ('gas', 'H_density')].in_units('msun/kpc**3')*current_a**3/h**2
  H_0_dens =  data_grid[ ('gas', 'H_p0_density')].in_units('msun/kpc**3')*current_a**3/h**2
  H_1_dens =  data_grid[ ('gas', 'H_p1_density')].in_units('msun/kpc**3')*current_a**3/h**2
  # He_dens =  data_grid[ ('gas', 'He_density')].in_units('msun/kpc**3')*current_a**3/h**2
  He_0_dens =  data_grid[ ('gas', 'He_p0_density')].in_units('msun/kpc**3')*current_a**3/h**2
  He_1_dens =  data_grid[ ('gas', 'He_p1_density')].in_units('msun/kpc**3')*current_a**3/h**2
  He_2_dens =  data_grid[ ('gas', 'He_p2_density')].in_units('msun/kpc**3')*current_a**3/h**2
  electron_dens =  data_grid[ ('gas', 'El_density')].in_units('msun/kpc**3')*current_a**3/h**2
  proten_electron_mass_ratio = 1836.15267389
  electron_dens *= proten_electron_mass_ratio

if metals:
  metal_dens = data_grid[ ('gas', 'metal_density')].in_units('msun/kpc**3')*current_a**3/h**2


# H_frac = H_0_dens / gas_dens
# He_frac = He_0_dens / gas_dens
#
# mu_0 = 1. / ( H_frac + He_frac/4 )
#


p_mass = data[('all', 'particle_mass')].in_units('msun')*h
p_pos_x = data[('all', 'particle_position_x')].in_units('kpc')/current_a*h
p_pos_y = data[('all', 'particle_position_y')].in_units('kpc')/current_a*h
p_pos_z = data[('all', 'particle_position_z')].in_units('kpc')/current_a*h
p_vel_x = data[('all', 'particle_velocity_x')].in_units('km/s')
p_vel_y = data[('all', 'particle_velocity_y')].in_units('km/s')
p_vel_z = data[('all', 'particle_velocity_z')].in_units('km/s')

data_enzo = { 'dm':{}, 'gas':{} }
data_enzo['current_a'] = current_a
data_enzo['current_z'] = current_z

data_enzo['dm']['mass'] = p_mass
data_enzo['dm']['pos_x'] = p_pos_x
data_enzo['dm']['pos_y'] = p_pos_y
data_enzo['dm']['pos_z'] = p_pos_z
data_enzo['dm']['vel_x'] = p_vel_x
data_enzo['dm']['vel_y'] = p_vel_y
data_enzo['dm']['vel_z'] = p_vel_z

data_enzo['gas']['density'] = gas_dens
data_enzo['gas']['momentum_x'] = gas_dens * gas_vel_x
data_enzo['gas']['momentum_y'] = gas_dens * gas_vel_y
data_enzo['gas']['momentum_z'] = gas_dens * gas_vel_z
data_enzo['gas']['GasEnergy'] = gas_u
data_enzo['gas']['Energy'] = gas_E
if cool:
  data_enzo['gas']['HI_density'] = H_0_dens
  data_enzo['gas']['HII_density'] = H_1_dens
  data_enzo['gas']['HeI_density'] = He_0_dens
  data_enzo['gas']['HeII_density'] = He_1_dens
  data_enzo['gas']['HeIII_density'] = He_2_dens
  data_enzo['gas']['e_density'] = electron_dens
if metals:
  data_enzo['gas']['metal_density'] = metal_dens


proc_grid = [ 2, 2, 2]
box_size = [ 50000, 50000, 50000 ]
grid_size = [ 128, 128, 128 ]
outputBaseName = '{0}_particles.h5'.format(nSnap)
generate_ics_particles(data_enzo, outputDir, outputBaseName, proc_grid, box_size, grid_size)

outputBaseName = '{0}.h5'.format(nSnap)
expand_data_grid_to_cholla( proc_grid, data_enzo['gas'], outputDir, outputBaseName )
