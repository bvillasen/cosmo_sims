import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import h5py as h5
from power_spectrum import get_power_spectrum
from internal_energy import get_temp

dataDir = '/raid/bruno/data/'
# dataDir = '/home/bruno/Desktop/data/'
dev_dir = '/home/bruno/Desktop/Dropbox/Developer/'
cosmo_dir = dev_dir + 'cosmo_sims/'
toolsDirectory = cosmo_dir + "tools/"
sys.path.extend([toolsDirectory ] )
from tools import *
from load_data_cholla import load_snapshot_data
from load_data_nyx import load_snapshot_nyx
from load_data_enzo import load_snapshot_enzo
from load_data_ramses import load_snapshot_ramses

n_cholla_files = 3

enzoDir = dataDir + 'cosmo_sims/enzo/128_cool_uv_50Mpc_HLLC_grav4/h5_files/'
chollaDir = dataDir + 'cosmo_sims/cholla_pm/128_cool_uv_50Mpc/'

data_name_0 = 'SIMPLE_PPMP_eta0.005_beta0.00_grav4'
data_name_1 = 'SIMPLE_PPMP_eta0.010_beta0.00_grav4'
data_name_2 = 'SIMPLE_PPMP_eta0.035_beta0.00_grav4'

outputsDir = '/home/bruno/cholla/scale_output_files/'
ramsesDir = dataDir + 'cosmo_sims/ramses/128_hydro_50Mpc_slope1_cfl2/h5_files/'
chollaDir_0 = chollaDir + 'data_{0}/'.format( data_name_0)
chollaDir_1 = chollaDir + 'data_{0}/'.format( data_name_1)
chollaDir_2 = chollaDir + 'data_{0}/'.format( data_name_2)

chollaDir_all = [ chollaDir_0, chollaDir_1, chollaDir_2 ]

labels = [ r'Cholla $\eta=0.005$', r'Cholla $\eta=0.010$', r'Cholla $\eta=0.035$',]

outDir = dev_dir + 'figures/comparison_enzo/'

create_directory( outDir )

fileName = 'temperature_comparison_enzo_ramses_uv.png'

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



z_list_rm = []
t_rm_list = []

for nSnap in range(29):
  
  #Load Ramses data
  data_ramses = load_snapshot_ramses( nSnap, ramsesDir, dm=True, particles=False, cool=False, metals=False, hydro=True)
  current_a_ramses = data_ramses['current_a']
  current_z_ramses = data_ramses['current_z']
  temp_ramses = data_ramses['gas']['temperature'][...]
  print ' Ramses: ', current_z_ramses 
  t_rm = temp_ramses.mean()
  z_list_rm.append(current_z_ramses)
  t_rm_list.append(t_rm)

z_list = []
t_ch_list_all = []
t_en_list = []
t_cosmo_list = []


nSnap = 0

for i in range( n_cholla_files):
  t_ch_list = []
  for nSnap in range(28):
    data_cholla = load_snapshot_data( nSnap, chollaDir_all[i], single_file=False )
    current_z_ch = data_cholla['current_z']
    gas_dens_ch = data_cholla['gas']['density'][...]
    gas_U_ch = data_cholla['gas']['GasEnergy'][...] / gas_dens_ch
    temp_ch = get_temp( gas_U_ch*1e6)
    print ' Cholla: ', current_z_ch
    t_ch = temp_ch.mean()
    t_ch_list.append( t_ch )
  t_ch_list_all.append( t_ch_list)

for nSnap in range(28):
  #Load Ramses data
  data_enzo = load_snapshot_enzo( nSnap, enzoDir, dm=True, particles=False, cool=False, metals=False, hydro=True)
  current_a_enzo = data_enzo['current_a']
  current_z_enzo = data_enzo['current_z']
  gas_dens_en = data_enzo['gas']['density'][...]
  gas_U_en = data_enzo['gas']['GasEnergy'][...] / gas_dens_en
  temp_enzo = get_temp( gas_U_en*1e6)
  print temp_enzo.min()
  # temp_enzo = data_enzo['gas']['temperature'][...]
  
  print ' Enzo: ', current_z_enzo 

  t_en = temp_enzo.mean()
  
  
  if nSnap == 0:
    t0 = t_en
    a_0 = current_a_enzo
  
  a = current_a_enzo
  t_cosmo = t0 * a_0**2 / a**2
  
  # t_rm *=  a_0**2 / a**2

  z_list.append( current_z_enzo)
  t_en_list.append( t_en )
  t_cosmo_list.append(t_cosmo)


fig = plt.figure(0)
# fig.set_size_inches(20,12)
fig.clf()
ax = plt.gca()


fs = 15
for i in range(n_cholla_files):
  label = labels[i]
  ax.plot( z_list, t_ch_list_all[i], linewidth=5, label=label)
ax.plot( z_list, t_en_list, linewidth=3, label='Enzo')
# ax.plot( z_list_rm, t_rm_list, linewidth=3, label='Ramses')
ax.plot( z_list, t_cosmo_list, '--', label='Cosmo')
ax.legend(fontsize=10)

ax.set_yscale('log')
ax.set_xlim( 0, 100 )
ax.set_ylim( 0.1, 3e5 )

# ax.set_title()
ax.set_xlabel('Redshift', fontsize=fs)
ax.set_ylabel('Avrg Temperature [k]', fontsize=fs)

fig.savefig( outDir + fileName, ppi=200 )
