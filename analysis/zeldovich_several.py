import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import yt

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

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# rank = 0


dataDir = '/raid/bruno/data/'
# enzoDir = dataDir + 'cosmo_sims/enzo/ZeldovichPancake/'


data_set = 'test'
inputDir_0 = dataDir + 'cosmo_sims/cholla_pm/zeldovich/data_de0_noGrav_da00001/'
inputDir_1 = dataDir + 'cosmo_sims/cholla_pm/zeldovich/data_de0_noGrav_da0001_velTimes2_SIMPLE_PCM/'
inputDir_2 = dataDir + 'cosmo_sims/cholla_pm/zeldovich/data_de0_noGrav_da0001_velTimes2_SIMPLE_PLMP/'
inputDir_3 = dataDir + 'cosmo_sims/cholla_pm/zeldovich/data_de0_noGrav_da0001_velTimes2_SIMPLE_PLMC/'
inputDir_4 = dataDir + 'cosmo_sims/cholla_pm/zeldovich/data_de0_noGrav_da0001_velTimes2_SIMPLE_PPMC/'
# inputDir_5 = dataDir + 'cosmo_sims/cholla_pm/zeldovich/data_de0_noGrav_da00001_velTimes2_HLL_PCM/'



outputDir = cosmo_dir + 'figures/zeldovich/energy_da00001_HLL_SIMPLE/'



if rank == 0:
  # print 'Input Dir: ', inputDir
  print 'Output Dir: ', outputDir
  create_directory( outputDir )

rho_0 = 277.535989
v_factor_0 = 0.1



DUAL_ENERGY =   False

a_list = []

gamma = 5./3

j_indx = 0
i_indx = 0

L = 64.
n = 256
dx = L / ( n )
x = np.arange(0, 256, 1)* dx + 0.5*dx

nSnap = rank 
# for nSnap in range( 80 ):

out_file_name = 'zeldovich_{0}.png'.format(nSnap)


data_cholla = load_snapshot_data( nSnap, inputDir_0 )
current_z_ch = data_cholla['current_z']
current_a = data_cholla['current_a']
dens_dm_cholla = data_cholla['dm']['density'][...]
dens_ch = data_cholla['gas']['density'][...][:, j_indx, i_indx]
vel_x_ch = data_cholla['gas']['momentum_x'][...][:, j_indx, i_indx] / dens_ch
vel_y_ch = data_cholla['gas']['momentum_y'][...][:, j_indx, i_indx] / dens_ch
vel_z_ch = data_cholla['gas']['momentum_z'][...][:, j_indx, i_indx] / dens_ch
E_ch = data_cholla['gas']['Energy'][...][:, j_indx, i_indx]
Ekin = 0.5 * dens_ch * ( vel_x_ch*vel_x_ch + vel_y_ch*vel_y_ch + vel_z_ch*vel_z_ch )
U_ch = E_ch - Ekin
if DUAL_ENERGY: U_ch = data_cholla['gas']['GasEnergy'][...][:, j_indx, i_indx]
temp_ch = get_temp(U_ch / dens_ch * 1e6, mu=1)

# 
# rho_factor = 1 / rho_0
# mom_factor = 1 / rho_0 / v_factor_0 * current_a
# v_factor = 1 / v_factor_0 * current_a
# e_factor = 1 / rho_0 / v_factor_0 / v_factor_0 * current_a * current_a
# 
d_0, v_0, E_0, Ekin_0, U_0 = dens_ch, vel_x_ch, E_ch, Ekin, U_ch

data_cholla = load_snapshot_data( nSnap, inputDir_1 )
current_z_ch = data_cholla['current_z']
dens_dm_cholla = data_cholla['dm']['density'][...]
dens_ch = data_cholla['gas']['density'][...][:, j_indx, i_indx]
vel_x_ch = data_cholla['gas']['momentum_x'][...][:, j_indx, i_indx] / dens_ch
vel_y_ch = data_cholla['gas']['momentum_y'][...][:, j_indx, i_indx] / dens_ch
vel_z_ch = data_cholla['gas']['momentum_z'][...][:, j_indx, i_indx] / dens_ch
E_ch = data_cholla['gas']['Energy'][...][:, j_indx, i_indx]
Ekin = 0.5 * dens_ch * ( vel_x_ch*vel_x_ch + vel_y_ch*vel_y_ch + vel_z_ch*vel_z_ch )
U_ch = E_ch - Ekin
if DUAL_ENERGY: U_ch = data_cholla['gas']['GasEnergy'][...][:, j_indx, i_indx]
temp_ch = get_temp(U_ch / dens_ch * 1e6, mu=1)

d_1, v_1, E_1, Ekin_1, U_1 = dens_ch, vel_x_ch, E_ch, Ekin, U_ch

data_cholla = load_snapshot_data( nSnap, inputDir_2 )
current_z_ch = data_cholla['current_z']
dens_dm_cholla = data_cholla['dm']['density'][...]
dens_ch = data_cholla['gas']['density'][...][:, j_indx, i_indx]
vel_x_ch = data_cholla['gas']['momentum_x'][...][:, j_indx, i_indx] / dens_ch
vel_y_ch = data_cholla['gas']['momentum_y'][...][:, j_indx, i_indx] / dens_ch
vel_z_ch = data_cholla['gas']['momentum_z'][...][:, j_indx, i_indx] / dens_ch
E_ch = data_cholla['gas']['Energy'][...][:, j_indx, i_indx]
Ekin = 0.5 * dens_ch * ( vel_x_ch*vel_x_ch + vel_y_ch*vel_y_ch + vel_z_ch*vel_z_ch )
U_ch = E_ch - Ekin
if DUAL_ENERGY: U_ch = data_cholla['gas']['GasEnergy'][...][:, j_indx, i_indx]
temp_ch = get_temp(U_ch / dens_ch * 1e6, mu=1)

d_2, v_2, E_2, Ekin_2, U_2 = dens_ch, vel_x_ch, E_ch, Ekin, U_ch
# 
data_cholla = load_snapshot_data( nSnap, inputDir_3 )
current_z_ch = data_cholla['current_z']
dens_dm_cholla = data_cholla['dm']['density'][...]
dens_ch = data_cholla['gas']['density'][...][:, j_indx, i_indx]
vel_x_ch = data_cholla['gas']['momentum_x'][...][:, j_indx, i_indx] / dens_ch
vel_y_ch = data_cholla['gas']['momentum_y'][...][:, j_indx, i_indx] / dens_ch
vel_z_ch = data_cholla['gas']['momentum_z'][...][:, j_indx, i_indx] / dens_ch
E_ch = data_cholla['gas']['Energy'][...][:, j_indx, i_indx]
Ekin = 0.5 * dens_ch * ( vel_x_ch*vel_x_ch + vel_y_ch*vel_y_ch + vel_z_ch*vel_z_ch )
U_ch = E_ch - Ekin
if DUAL_ENERGY: U_ch = data_cholla['gas']['GasEnergy'][...][:, j_indx, i_indx]
temp_ch = get_temp(U_ch / dens_ch * 1e6, mu=1)

d_3, v_3, E_3, Ekin_3, U_3 = dens_ch, vel_x_ch, E_ch, Ekin, U_ch

data_cholla = load_snapshot_data( nSnap, inputDir_4 )
current_z_ch = data_cholla['current_z']
dens_dm_cholla = data_cholla['dm']['density'][...]
dens_ch = data_cholla['gas']['density'][...][:, j_indx, i_indx]
vel_x_ch = data_cholla['gas']['momentum_x'][...][:, j_indx, i_indx] / dens_ch
vel_y_ch = data_cholla['gas']['momentum_y'][...][:, j_indx, i_indx] / dens_ch
vel_z_ch = data_cholla['gas']['momentum_z'][...][:, j_indx, i_indx] / dens_ch
E_ch = data_cholla['gas']['Energy'][...][:, j_indx, i_indx]
Ekin = 0.5 * dens_ch * ( vel_x_ch*vel_x_ch + vel_y_ch*vel_y_ch + vel_z_ch*vel_z_ch )
U_ch = E_ch - Ekin
if DUAL_ENERGY: U_ch = data_cholla['gas']['GasEnergy'][...][:, j_indx, i_indx]
temp_ch = get_temp(U_ch / dens_ch * 1e6, mu=1)

d_4, v_4, E_4, Ekin_4, U_4 = dens_ch, vel_x_ch, E_ch, Ekin, U_ch
# 
# data_cholla = load_snapshot_data( nSnap, inputDir_5 )
# current_z_ch = data_cholla['current_z']
# dens_dm_cholla = data_cholla['dm']['density'][...]
# dens_ch = data_cholla['gas']['density'][...][:, j_indx, i_indx]
# vel_x_ch = data_cholla['gas']['momentum_x'][...][:, j_indx, i_indx] / dens_ch
# vel_y_ch = data_cholla['gas']['momentum_y'][...][:, j_indx, i_indx] / dens_ch
# vel_z_ch = data_cholla['gas']['momentum_z'][...][:, j_indx, i_indx] / dens_ch
# E_ch = data_cholla['gas']['Energy'][...][:, j_indx, i_indx]
# Ekin = 0.5 * dens_ch * ( vel_x_ch*vel_x_ch + vel_y_ch*vel_y_ch + vel_z_ch*vel_z_ch )
# U_ch = E_ch - Ekin
# if DUAL_ENERGY: U_ch = data_cholla['gas']['GasEnergy'][...][:, j_indx, i_indx]
# temp_ch = get_temp(U_ch / dens_ch * 1e6, mu=1)
# 
# d_5, v_5, E_5, Ekin_5, U_5 = dens_ch, vel_x_ch, E_ch, Ekin, U_ch


n_rows = 5
n_cols = 1
fig, ax_list = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(10*n_cols,3*n_rows))

text = ' z = {0:.02f}'.format(current_z_ch)
props = dict(boxstyle='round', facecolor='gray', alpha=0.3)

ax = ax_list[0]
# ax.plot( x, gas_dens, linewidth=3, label='Enzo'  )
ax.plot( x, d_0, linewidth=3, label='Total No Grav'    )
ax.plot( x, d_1,   label='Total No Grav vel X 2 HLLC PCM'  )
ax.plot( x, d_2,  label='Total No Grav vel X 2 HLLC PLMP'  )
ax.plot( x, d_3,  label='Total No Grav vel X 2 HLLC PLMC'  )
ax.plot( x, d_4,  label='Total No Grav vel X 2 HLLC PPMC'  )
# ax.plot( x, d_5,  label='Total No Grav vel X 2 HLLC PCM'  )
ax.set_yscale('log')
ax.set_xlim(0,64)
ax.set_ylabel(r'Density  [ $h^3 M_{\odot}  kpc^{-3} $]')
ax.text(0.02, 0.9, text, transform=ax.transAxes, fontsize=14,
          verticalalignment='top', bbox=props)
ax.legend(loc=0)


ax = ax_list[1]
# ax.plot( x, gas_vel, linewidth=3 )
ax.plot( x, v_0, linewidth=3 )
ax.plot( x, v_1,  )
ax.plot( x, v_2 )
ax.plot( x, v_3 )
ax.plot( x, v_4 )
# ax.plot( x, v_5 )
ax.set_xlim(0,64)
ax.set_ylabel(r'Velocity  [ $km/s$ ]')
ax.set_xlabel(r'X [ $\mathrm{cMpc}/h$ ]')

ax = ax_list[2]
# ax.plot( x, gas_temp, linewidth=3  )
ax.plot( x, U_0, linewidth=3 )
ax.plot( x, U_1, )
ax.plot( x, U_2 )
ax.plot( x, U_3 )
ax.plot( x, U_4 )
# ax.plot( x, U_5 )
ax.set_yscale('log')
ax.set_xlim(0,64)
# ax.set_ylim(40,200)
ax.set_ylabel(r'Thermal Energy')

ax = ax_list[3]
# ax.plot( x, gas_temp, linewidth=3  )
ax.plot( x, E_0, linewidth=3  )
ax.plot( x, E_1,   )
ax.plot( x, E_2  )
ax.plot( x, E_3  )
ax.plot( x, E_4  )
# ax.plot( x, E_5  )
ax.set_yscale('log')
ax.set_xlim(0,64)
# ax.set_ylim(40,200)
ax.set_ylabel(r'Energy')

ax = ax_list[4]
# ax.plot( x, gas_temp, linewidth=3  )
ax.plot( x, Ekin_0, linewidth=3  )
ax.plot( x, Ekin_1,  )
ax.plot( x, Ekin_2  )
ax.plot( x, Ekin_3  )
ax.plot( x, Ekin_4  )
# ax.plot( x, Ekin_5  )
ax.set_yscale('log')
ax.set_xlim(0,64)
# ax.set_ylim(40,200)
ax.set_ylabel(r'Kinetic Energy')


fig.tight_layout()
fig.savefig( outputDir + out_file_name)
print "Saved Fig: ", outputDir + out_file_name


# np.savetxt('outputs_zeldovich.txt', a_list )
