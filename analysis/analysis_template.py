import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5

dataDir = '/home/bruno/Desktop/data/'
cosmo_dir = '/home/bruno/Desktop/Dropbox/Developer/cosmo_sims/'
outDir = cosmo_dir + 'figures/phase_diagram/'
toolsDirectory = cosmo_dir + "tools/"
sys.path.extend([toolsDirectory ] )
from load_data_cholla import load_snapshot_data
from load_data_enzo import load_snapshot_enzo
from internal_energy import get_internal_energy, get_temp, get_Temperaure_From_Flags_DE
from cosmo_constants import *


dataDir = '/raid/bruno/data/'
enzoDir = dataDir + 'cosmo_sims/enzo/256_cool_uv/h5_files/'


eta_0 = 0.001
beta_0 = 0.05
beta_1 = 0.00
chollaDir = dataDir + 'cosmo_sims/cholla_pm/256_cool/data_PPMC_HLLC_SIMPLE_eta{0:.3f}_beta{1:.3f}_{2:.3f}/'.format( eta_0, beta_0, beta_1 )


nSnap = 0  #Snapshot Number [0 - 30]


data_cholla = load_snapshot_data( nSnap, chollaDir, cool=True)
H0 = data_cholla['H0'] #[km/s / kpc]
h = H0 * 10
Omega_M = data_cholla['Omega_M']
Omega_L = data_cholla['Omega_L']
current_z =  data_cholla['current_z']
dens_dm = data_cholla['dm']['density'][...] 
dens = data_cholla['gas']['density'][...]
dens_mean = dens.mean()
temp = data_cholla['gas']['temperature'][...]
flags_DE = data_cholla['gas']['flags_DE'][...]        #1 if using Advected Internal Energy, 0 otherwise
HI_dens = data_cholla['gas']['HI_density'][...]
HII_dens = data_cholla['gas']['HII_density'][...]


data_enzo = load_snapshot_enzo( nSnap, enzoDir, dm=True, cool=True)
current_z_enzo = data_enzo['current_z']
dens_dm_enzo = data_enzo['dm']['density'][...]
dens_gas_enzo = data_enzo['gas']['density'][...]
dens_gas_H_enzo = data_enzo['gas']['HI_density'][...]
dens_gas_HII_enzo = data_enzo['gas']['HII_density'][...]
