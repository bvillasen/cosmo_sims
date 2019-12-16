import os, sys
from os import listdir
from os.path import isfile, join
import h5py as h5
import numpy as np

currentDirectory = os.getcwd()
#Add Modules from other directories
cosmo_dir = '/home/bruno/Desktop/Dropbox/Developer/cosmo_sims/'
outDir = cosmo_dir + 'figures/power_hydro/'
toolsDirectory = cosmo_dir + "tools/"
sys.path.extend([toolsDirectory ] )
from load_data_nyx import load_snapshot_nyx
from generate_ics_particles_functions import generate_ics_particles, generate_ics_particles_single_domain
from expand_data_grid import expand_data_grid_to_cholla
from domain_decomposition import get_domain_block, get_domain_parent
import ast


dataDir = '/raid/bruno/data/'
nyxDir = dataDir + 'cosmo_sims/nyx/256_dm_50Mpc/'
outDir = dataDir + 'cosmo_sims/cholla_pm/256_dm_50Mpc/ics_nyx/ics_4/'
nSnap = 0




data_nyx = load_snapshot_nyx( nSnap, nyxDir, hydro=False )

Lbox = 50000

proc_grid = [ 2, 2, 1]
box_size = [ Lbox, Lbox, Lbox ]
grid_size = [ 256, 256, 256 ]
domain_parent = get_domain_block( proc_grid, box_size, grid_size )
outputBaseName = '{0}_particles.h5'.format(nSnap)
generate_ics_particles(data_nyx, outDir, outputBaseName, proc_grid, box_size, grid_size)
