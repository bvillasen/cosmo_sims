import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import yt

dev_dir = '/home/bruno/Desktop/Dropbox/Developer/'
cosmo_dir = dev_dir + 'cosmo_sims/'
toolsDirectory = cosmo_dir + "tools/"
sys.path.extend([toolsDirectory ] )
from load_data_cholla import load_snapshot_data
from internal_energy import get_internal_energy, get_temp, get_Temperaure_From_Flags_DE
# from load_data_enzo import load_snapshot_enzo
from cosmo_constants import *
from tools import create_directory

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nSnap = rank

# rank = 0


dataDir = '/raid/bruno/data/'
# dataDir = '/home/bruno/Desktop/data/'

data_set = 'enzo_simple_beta_convDE'
startSnap = 27




enzoDir = dataDir + 'cosmo_sims/enzo/ZeldovichPancake_HLLC/'


outDir = dev_dir + 'figures/zeldovich_mass/'
if rank == 0:
  create_directory( outDir )

a_list = []

gamma = 5./3

j_indx = 0
i_indx = 0

L = 64.
n = 256
dx = L / ( n )
x = np.arange(0, 256, 1)* dx + 0.5*dx
dv = (dx*1e3)**3

chollaDir_0 = dataDir + 'cosmo_sims/cholla_pm/zeldovich/data_PLMC_HLLC_VL_eta0.001_0.030_z1/'
chollaDir_1 = dataDir + 'cosmo_sims/cholla_pm/zeldovich/data_PLMP_HLLC_VL_eta0.001_0.030_z1/'
chollaDir_2 = dataDir + 'cosmo_sims/cholla_pm/zeldovich/data_PPMC_HLLC_VL_eta0.001_0.030_z1_ic0/'
chollaDir_3 = dataDir + 'cosmo_sims/cholla_pm/zeldovich/data_PPMP_HLLC_VL_eta0.001_0.030_z1_ic64/'
chollaDir_4 = dataDir + 'cosmo_sims/cholla_pm/zeldovich/data_PPMP_HLLC_VL_eta0.001_0.030_z1_ic32/'
chollaDir_5 = dataDir + 'cosmo_sims/cholla_pm/zeldovich/data_PPMP_HLLC_VL_eta0.001_0.030_z1_ic4/'
chollaDir_6 = dataDir + 'cosmo_sims/cholla_pm/zeldovich/data_PPMP_HLLC_VL_eta0.001_0.030_z1_ic0/'

# chollaDir_3 = dataDir + 'cosmo_sims/cholla_pm/zeldovich/data_PPMC_HLLC_VL_eta0.001_0.030_z1_signStone/'

dir_list = [ chollaDir_0, chollaDir_1, chollaDir_2, chollaDir_3, chollaDir_4, chollaDir_5, chollaDir_6  ]
labels = ['PLMC', 'PLMP', 'PPMC_ic0',  'PPMP_ic64', 'PPMP_ic32', 'PPMP_ic4', 'PPMP_ic0',  ]


out_file_name = 'zeldovich_mass.png'


#Plot UVB uvb_rates
nrows=1
ncols = 1
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10*ncols,8*nrows))

lw = 3

for i,chollaDir in enumerate(dir_list):
  print chollaDir
  mass = [] 
  z = []
  for nSnap in range(50):
    data_cholla = load_snapshot_data( nSnap, chollaDir )
    current_z = data_cholla['current_z']
    dens_ch = data_cholla['gas']['density'][...]
    mass_tot = dens_ch.sum() / dv
    z.append(current_z)
    mass.append( mass_tot )
  # print mass
  ax.plot( z, mass, label=labels[i] )
  
ax.legend()
ax.set_xlabel('Redshift')
ax.set_ylabel(r'Mass  [$\mathrm{M}_{\odot}/h$ ]')
fig.savefig( outDir+out_file_name,  bbox_inches='tight', dpi=100)