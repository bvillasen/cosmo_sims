import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5


# dataDir = '/raid/bruno/data/cosmo_sims/'
dataDir = '/home/bruno/Desktop/data/'
cosmo_dir = '/home/bruno/Desktop/Dropbox/Developer/cosmo_sims/'
toolsDirectory = cosmo_dir + "tools/"
sys.path.extend([toolsDirectory ] )
from load_data_cholla import load_snapshot_data
from internal_energy import get_internal_energy, get_temp, get_Temperaure_From_Flags_DE
# from load_data_enzo import load_snapshot_enzo
from cosmo_constants import *
from tools import create_directory


dataDir = '/raid/bruno/data/'
inputDir = dataDir + 'cosmo_sims/cholla_pm/uniform/data/'

# outputDir = cosmo_dir + 'figures/zeldovich/{0}/'.format( data_set )

nSnap = 0

data_cholla = load_snapshot_data( nSnap, inputDir, cosmo=False )

dens_dm = data_cholla['dm']['density'][...]
dens = data_cholla['gas']['density'][...]
vel_x = data_cholla['gas']['momentum_x'][...]
vel_y = data_cholla['gas']['momentum_y'][...]
vel_z = data_cholla['gas']['momentum_z'][...]
E = data_cholla['gas']['Energy'][...]
U = data_cholla['gas']['GasEnergy'][...]
temp = get_temp(U / dens * 1e6, mu=1)

