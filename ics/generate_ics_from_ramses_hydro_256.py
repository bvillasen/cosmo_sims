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
# from load_data_ramses_old import load_snapshot_ramses, load_snapshot_ramses_yt
from expand_data_grid import expand_data_grid_to_cholla
from expand_data_particles import expand_data_particles_to_cholla
from generate_ics_particles_functions import generate_ics_particles, generate_ics_particles_single_domain
from expand_data_grid import expand_data_grid_to_cholla
from domain_decomposition import get_domain_block, get_domain_parent
from tools import create_directory
from load_data_ramses import load_snapshot_ramses

# dataDir = '/home/bruno/Desktop/data/'
dataDir = '/raid/bruno/data/'
ramsesDir = dataDir + 'cosmo_sims/ramses/256_hydro_50Mpc/h5_files/'
inDir = ramsesDir
outputDir = dataDir + 'cosmo_sims/cholla_pm/256_hydro_50Mpc/ics_ramses/'
create_directory( outputDir )
nSnap_ramses = 0

cool = False
metals = False

nSnap = nSnap_ramses

data_ramses = load_snapshot_ramses( nSnap, inDir, dm=False, particles=True, cool=False, metals=False, hydro=True )

gas_dens = data_ramses['gas']['density'][...]
print gas_dens.mean(), gas_dens.min(), gas_dens.max(), 
# 
# proc_grid = [ 2, 2, 2]
# box_size = [ 50000, 50000, 50000 ]
# grid_size = [ 256, 256, 256 ]
# outputBaseName = '{0}_particles.h5'.format(nSnap)
# generate_ics_particles(data_ramses, outputDir, outputBaseName, proc_grid, box_size, grid_size)
# 
# outputBaseName = '{0}.h5'.format(nSnap)
# expand_data_grid_to_cholla( proc_grid, data_ramses['gas'], outputDir, outputBaseName )
