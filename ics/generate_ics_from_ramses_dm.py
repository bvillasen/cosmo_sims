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
from load_data_ramses import load_snapshot_ramses
from generate_ics_particles_functions import generate_ics_particles, generate_ics_particles_single_domain
from expand_data_grid import expand_data_grid_to_cholla
from domain_decomposition import get_domain_block, get_domain_parent
import ast
import yt


# dataDir = '/raid/bruno/data/'
dataDir = '/home/bruno/Desktop/data/'

inDir = dataDir + 'cosmo_sims/ramses/256_dm_50Mpc/h5_files/'
outDir = dataDir + 'cosmo_sims/cholla_pm/256_dm_50Mpc/ics_ramses/'



nSnap = 0

data_ramses = load_snapshot_ramses( nSnap, inDir, dm=True, cool=False, metals=False, hydro=False, particles=True  )
# 
Lbox = 50000
nPoints = 256

proc_grid = [ 1, 1, 1]
box_size = [ Lbox, Lbox, Lbox ]
grid_size = [ nPoints, nPoints, nPoints ]
domain_parent = get_domain_block( proc_grid, box_size, grid_size )
outputBaseName = '{0}_particles.h5'.format(nSnap)
generate_ics_particles(data_ramses, outDir, outputBaseName, proc_grid, box_size, grid_size)
