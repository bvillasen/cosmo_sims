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


n_cholla_files = 1
chollaDir_0 = dataDir + 'cosmo_sims/cholla_pm/zeldovich/data_PPMP_HLLC_VL_eta0.001_0.010/'
chollaDir_1 = dataDir + 'cosmo_sims/cholla_pm/zeldovich/data_PPMC_HLL_VL_eta0.001_0.030/'
chollaDir_2 = dataDir + 'cosmo_sims/cholla_pm/zeldovich/data_PPMC_HLLC_VL_eta0.001_0.050/'

chollaDir_all = [ chollaDir_0, chollaDir_1, chollaDir_2 ]
cholla_label_all = [r'Cholla PPMP', r'Cholla HLL',  r'Cholla $\eta_2=0.050$' ]



enzoDir = dataDir + 'cosmo_sims/enzo/ZeldovichPancake_HLLC/'
enzoDir_1 = dataDir + 'cosmo_sims/enzo/ZeldovichPancake_HLLC_noDE/'


outDir = dev_dir + 'figures/zeldovich_PPMP_eta0.010/'
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

# nSnap = 0
# for nSnap in range(  79 ):
out_file_name = 'zeldovich_{0}.png'.format( nSnap )

data_cholla_all = []

for n in range(n_cholla_files):
  chollaDir = chollaDir_all[n]
  data_cholla = load_snapshot_data( nSnap, chollaDir )
  current_z = data_cholla['current_z']
  dens_dm_cholla = data_cholla['dm']['density'][...]
  dens_ch = data_cholla['gas']['density'][...][:, j_indx, i_indx]
  vel_x_ch = data_cholla['gas']['momentum_x'][...][:, j_indx, i_indx] / dens_ch
  vel_y_ch = data_cholla['gas']['momentum_y'][...][:, j_indx, i_indx] / dens_ch
  vel_z_ch = data_cholla['gas']['momentum_z'][...][:, j_indx, i_indx] / dens_ch
  E_ch = data_cholla['gas']['Energy'][...][:, j_indx, i_indx]
  U_ch = data_cholla['gas']['GasEnergy'][...][:, j_indx, i_indx]
  Ekin_ch = 0.5 * dens_ch * ( vel_x_ch*vel_x_ch + vel_y_ch*vel_y_ch + vel_z_ch*vel_z_ch)
  temp_ch = get_temp(U_ch / dens_ch * 1e6, mu=1)
  data_ch = [ dens_ch, vel_x_ch, temp_ch, U_ch, E_ch, Ekin_ch ]
  data_cholla_all.append(data_ch)



file_name = enzoDir + 'DD{0:04}/data{0:04}'.format(nSnap)
ds = yt.load( file_name )
data = ds.all_data()
h = ds.hubble_constant
current_z = ds.current_redshift
current_a = 1./(current_z + 1)
x = data[('gas', 'x')].in_units('Mpc/h').v / current_a
gas_dens = data[ ('gas', 'density')].in_units('msun/kpc**3').v*current_a**3/h**2
gas_temp = data[ ('gas', 'temperature')].v
gas_vel = data[ ('gas', 'velocity_x')].in_units('km/s').v
gas_u = data[('gas', 'thermal_energy' )].v * 1e-10 *gas_dens #km^2/s^2
mu = data[('gas', 'mean_molecular_weight' )]
temp = get_temp(gas_u / gas_dens * 1e6, mu=mu)
Ekin = 0.5 * gas_dens * gas_vel * gas_vel
gas_E = Ekin + gas_u
data_en = [ gas_dens, gas_vel, temp,  gas_u, gas_E, Ekin ]
# 
# file_name = enzoDir_1 + 'DD{0:04}/data{0:04}'.format(nSnap)
# ds = yt.load( file_name )
# data = ds.all_data()
# h = ds.hubble_constant
# current_z = ds.current_redshift
# current_a = 1./(current_z + 1)
# x = data[('gas', 'x')].in_units('Mpc/h').v / current_a
# gas_dens = data[ ('gas', 'density')].in_units('msun/kpc**3').v*current_a**3/h**2
# gas_temp = data[ ('gas', 'temperature')].v
# gas_vel = data[ ('gas', 'velocity_x')].in_units('km/s').v
# gas_u = data[('gas', 'thermal_energy' )].v * 1e-10 *gas_dens #km^2/s^2
# mu = data[('gas', 'mean_molecular_weight' )]
# temp = get_temp(gas_u / gas_dens * 1e6, mu=mu)
# Ekin = 0.5 * gas_dens * gas_vel * gas_vel
# gas_E = Ekin + gas_u
# data_en_1 = [ gas_dens, gas_vel, temp,  gas_u, gas_E, Ekin ]
# 

n_rows = 3
n_cols = 1
fig, ax_list = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(10*n_cols,3*n_rows))

text = ' z = {0:.02f}'.format(current_z)
props = dict(boxstyle='round', facecolor='gray', alpha=0.3)




ax = ax_list[0]
ax.plot( x, data_en[0], linewidth=3, label='Enzo'    )
# ax.plot( x, data_en_1[0], linewidth=1, label='Enzo_HLLC noDE'    )


ax = ax_list[1]
ax.plot( x, data_en[1], linewidth=3 )
# ax.plot( x, data_en_1[1], linewidth=1 )

ax = ax_list[2]
ax.plot( x, data_en[2], linewidth=3 )
# ax.plot( x, data_en_1[2], linewidth=1 )


for n in range(n_cholla_files):
  data_ch = data_cholla_all[n]

  ax = ax_list[0]
  ax.plot( x, data_ch[0],  label=cholla_label_all[n] )

  ax = ax_list[1]
  ax.plot( x, data_ch[1] )

  ax = ax_list[2]
  ax.plot( x, data_ch[2] )


ax = ax_list[0]
ax.set_yscale('log')
ax.set_xlim(0,64)
ax.set_ylabel(r'Density  [ $h^3 M_{\odot}  kpc^{-3} $]')
ax.text(0.02, 0.9, text, transform=ax.transAxes, fontsize=14,
          verticalalignment='top', bbox=props)
ax.legend(loc=0)


ax = ax_list[1]
ax.set_xlim(0,64)
ax.set_ylabel(r'Velocity  [ $km/s$ ]')
ax.set_xlabel(r'X [ $\mathrm{cMpc}/h$ ]')

ax = ax_list[2]
ax.set_yscale('log')
ax.set_xlim(0,64)
ax.set_ylabel(r'Temperature  [K]')


fig.tight_layout()
fig.savefig( outDir + out_file_name)
print( "Saved image: " + outDir + out_file_name)


# np.savetxt('outputs_zeldovich.txt', a_list )
