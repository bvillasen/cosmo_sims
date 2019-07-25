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
from load_data_gadget import load_gadget_file, load_gadget_snapshot, load_data_particles
import ast



dataDir = '/raid/bruno/data/'
gadgetDir = dataDir + 'cosmo_sims/gadget/dm_256_100Mpc/'
outDir = dataDir + 'cosmo_sims/nyx/256_dm_100Mpc/'
nSnap = 0


data_gadget = load_gadget_snapshot( nSnap, gadgetDir, hydro=False, CIC=False )

h = 0.6766

mass = data_gadget['dm']['mass'] / h
pos_x = data_gadget['dm']['pos_x'] / h / 1000
pos_y = data_gadget['dm']['pos_y'] / h / 1000
pos_z = data_gadget['dm']['pos_z'] / h / 1000
vel_x = data_gadget['dm']['vel_x']
vel_y = data_gadget['dm']['vel_y']
vel_z = data_gadget['dm']['vel_z']
n_particles = len(mass)
# #
data_dm = np.array([ pos_x, pos_y, pos_z, mass, vel_x, vel_y, vel_z ]).T

outFileName = 'ic_256_dm_100Mpc.ascii'
# np.savetxt( outDir + outFileName, data_dm, header=str(n_particles), comments='' )
