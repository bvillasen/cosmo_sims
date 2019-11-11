import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import h5py as h5
from power_spectrum import get_power_spectrum
from internal_energy import get_temp

# dataDir = '/raid/bruno/data/'
dataDir = '/home/bruno/Desktop/data/'
dev_dir = '/home/bruno/Desktop/Dropbox/Developer/'
cosmo_dir = dev_dir + 'cosmo_sims/'
toolsDirectory = cosmo_dir + "tools/"
sys.path.extend([toolsDirectory ] )
from tools import *
from load_data_cholla import load_snapshot_data
from load_data_nyx import load_snapshot_nyx
from load_data_ramses import load_snapshot_ramses


outputsDir = '/home/bruno/cholla/scale_output_files/'
ramsesDir = dataDir + 'cosmo_sims/ramses/128_hydro_50Mpc_slope1/h5_files/'
chollaDir = dataDir + 'cosmo_sims/cholla_pm/128_hydro_50Mpc/data_ramses_PLMC_beta0.25_slope1/'
outDir = dev_dir + 'figures/comparison_ramses/'

create_directory( outDir )

fileName = 'temperature_comparison_beta0.25_slope1.png'

Lbox = 50.
h = 0.6766

nPoints = 128
nz, ny, nx = nPoints, nPoints, nPoints
nCells  = nx*ny*nz
Lx = Lbox
Ly = Lbox
Lz = Lbox
dx, dy, dz = Lx/(nx), Ly/(ny), Lz/(nz )
n_kSamples = 12





fig = plt.figure(0)
# fig.set_size_inches(20,12)
fig.clf()
ax = plt.gca()

z_list = []
t_ch_list = []
t_rm_list = []
t_cosmo_list = []

nSnap = 0
for nSnap in range(29):

  data_cholla = load_snapshot_data( nSnap, chollaDir, single_file=True )
  current_z_ch = data_cholla['current_z'][0]
  gas_dens_ch = data_cholla['gas']['density'][...]
  gas_U_ch = data_cholla['gas']['GasEnergy'][...] / gas_dens_ch
  temp_ch = get_temp( gas_U_ch*1e6)
  print ' Cholla: ', current_z_ch

  #Load Ramses data
  data_ramses = load_snapshot_ramses( nSnap, ramsesDir, dm=True, particles=False, cool=False, metals=False, hydro=True)
  current_a_ramses = data_ramses['current_a']
  current_z_ramses = data_ramses['current_z']
  temp_ramses = data_ramses['gas']['temperature'][...]
  print ' Ramses: ', current_z_ramses 

  t_ch = temp_ch.mean()
  t_rm = temp_ramses.mean()
  
  diff = ( t_ch - t_rm )/t_rm
  print diff
  
  if nSnap == 0:
    t0 = t_ch
    a_0 = current_a_ramses
  
  a = current_a_ramses
  t_cosmo = t0 * a_0**2 / a**2
  
  # t_rm *=  a_0**2 / a**2

  z_list.append( current_z_ch)
  t_ch_list.append( t_ch )
  t_rm_list.append( t_rm )
  t_cosmo_list.append(t_cosmo)


fs = 15
ax.plot( z_list, t_ch_list, linewidth=5, label='Cholla')
ax.plot( z_list, t_rm_list, linewidth=3, label='Ramses')
ax.plot( z_list, t_cosmo_list, '--', label='Cosmo')
ax.legend(fontsize=fs)

ax.set_yscale('log')
ax.set_xlim( 0, 100 )
ax.set_ylim( 0.1, 3e5 )

ax.set_xlabel('Redshift', fontsize=fs)
ax.set_ylabel('Avrg Temperature [k]', fontsize=fs)

fig.savefig( outDir + fileName )
