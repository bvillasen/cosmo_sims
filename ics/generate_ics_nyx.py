import os, sys
from os import listdir
from os.path import isfile, join
import h5py as h5
import numpy as np
import yt

currentDirectory = os.getcwd()
#Add Modules from other directories
cosmo_dir = '/home/bruno/Desktop/Dropbox/Developer/cosmo_sims/'
outDir = cosmo_dir + 'figures/power_hydro/'
toolsDirectory = cosmo_dir + "tools/"
sys.path.extend([toolsDirectory ] )
# from load_data_gadget import load_gadget_file, load_gadget_snapshot, load_data_particles
import ast



dataDir = '/home/bruno/Desktop/hard_drive_1/data/'
# gadgetDir = dataDir + 'cosmo_sims/gadget/dm_256_100Mpc/'
inDir = dataDir + 'cosmo_sims/enzo/256_dm_50Mpc/ics/'
outDir = dataDir + 'cosmo_sims/nyx/256_dm_50Mpc/'
nSnap = 0

snapKey = '{0:03}'.format(nSnap)
inFileName = 'DD0{0}/data0{0}'.format( snapKey)

ds = yt.load( inDir + inFileName )
data = ds.all_data()
h = ds.hubble_constant
current_z = np.float(ds.current_redshift)
current_a = 1./(current_z + 1)

mass = data[('all', 'particle_mass')].in_units('msun')
pos_x = data[('all', 'particle_position_x')].in_units('Mpc')/current_a
pos_y = data[('all', 'particle_position_y')].in_units('Mpc')/current_a
pos_z = data[('all', 'particle_position_z')].in_units('Mpc')/current_a
vel_x = data[('all', 'particle_velocity_x')].in_units('km/s')
vel_y = data[('all', 'particle_velocity_y')].in_units('km/s')
vel_z = data[('all', 'particle_velocity_z')].in_units('km/s')


# data_gadget = load_gadget_snapshot( nSnap, gadgetDir, hydro=False, CIC=False )
# h = 0.6766
# mass = data_gadget['dm']['mass'] / h
# pos_x = data_gadget['dm']['pos_x'] / h / 1000
# pos_y = data_gadget['dm']['pos_y'] / h / 1000
# pos_z = data_gadget['dm']['pos_z'] / h / 1000
# vel_x = data_gadget['dm']['vel_x']
# vel_y = data_gadget['dm']['vel_y']
# vel_z = data_gadget['dm']['vel_z']
# #

n_particles = len(mass)
data_dm = np.array([ pos_x, pos_y, pos_z, mass, vel_x, vel_y, vel_z ]).T

outFileName = 'ic_256_dm_50Mpc.ascii'
np.savetxt( outDir + outFileName, data_dm, header=str(n_particles), comments='' )
