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
nSnap = rank

# rank = 0


dataDir = '/raid/bruno/data/'
# enzoDir = dataDir + 'cosmo_sims/enzo/ZeldovichPancake/'


data_set = 'SIMPLE_PPMP_eta0.030_beta0.00_grav2'
inputDir = dataDir + 'cosmo_sims/cholla_pm/zeldovich/data_{0}/'.format(data_set)

outputDir = cosmo_dir + 'figures/zeldovich/{0}/'.format( data_set )

n_param = len( sys.argv )
if n_param > 1: 
  inputDir = sys.argv[1]
  outputDir = sys.argv[2]
  if inputDir[-1] != "/": inputDir += '/'
  if outputDir[-1] != "/": outputDir += '/'



if rank == 0:
  print 'Input Dir: ', inputDir
  print 'Output Dir: ', outputDir
  create_directory( outputDir )


DUAL_ENERGY =   False

a_list = []

gamma = 5./3

j_indx = 0
i_indx = 0

L = 64.
n = 256
dx = L / ( n )
x = np.arange(0, 256, 1)* dx + 0.5*dx

# nSnap = 0
# for nSnap in range( 80 ):

out_file_name = 'zeldovich_{0}.png'.format(nSnap)


data_cholla = load_snapshot_data( nSnap, inputDir )
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

# 
# rho_0 = dens_ch.mean()
# z_c = 1. #ZeldovichPancakeCollapseRedshift
# k_lambda = 64.
# T_init = 100.
# k = 2 * np.pi / k_lambda
# 
current_z = current_z_ch
h = 0.5
H_0 = 100 * h
# 
# rho = rho_0 / ( 1 - ( 1 + z_c )/( 1 + current_z ) * np.cos( k*( x - 32 )) ) 
# v = - H_0 * ( 1 + z_c )/np.sqrt( 1 + current_z ) * np.sin( k*( x - 32 )) / k
# T = T_init * ( rho / rho.mean() )**(2./3)

n_rows = 5
n_cols = 1
fig, ax_list = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(10*n_cols,3*n_rows))

text = ' z = {0:.02f}'.format(current_z)
props = dict(boxstyle='round', facecolor='gray', alpha=0.3)

ax = ax_list[0]
# ax.plot( x, gas_dens, linewidth=3, label='Enzo'  )
ax.plot( x, dens_ch,  label='Cholla'  )
# ax.plot( x, rho )
ax.set_yscale('log')
ax.set_xlim(0,64)
ax.set_ylabel(r'Density  [ $h^3 M_{\odot}  kpc^{-3} $]')
ax.text(0.02, 0.9, text, transform=ax.transAxes, fontsize=14,
          verticalalignment='top', bbox=props)
ax.legend(loc=0)


ax = ax_list[1]
# ax.plot( x, gas_vel, linewidth=3 )
ax.plot( x, vel_x_ch )
# ax.plot( x, v )
ax.set_xlim(0,64)
ax.set_ylabel(r'Velocity  [ $km/s$ ]')
ax.set_xlabel(r'X [ $\mathrm{cMpc}/h$ ]')

ax = ax_list[2]
# ax.plot( x, gas_temp, linewidth=3  )
ax.plot( x, U_ch )
# ax.plot( x, T  )
ax.set_yscale('log')
ax.set_xlim(0,64)
# ax.set_ylim(40,200)
ax.set_ylabel(r'Thermal Energy')

ax = ax_list[3]
# ax.plot( x, gas_temp, linewidth=3  )
ax.plot( x, E_ch  )
# ax.plot( x, T  )
ax.set_yscale('log')
ax.set_xlim(0,64)
# ax.set_ylim(40,200)
ax.set_ylabel(r'Energy')

ax = ax_list[4]
# ax.plot( x, gas_temp, linewidth=3  )
ax.plot( x, Ekin  )
# ax.plot( x, T  )
ax.set_yscale('log')
ax.set_xlim(0,64)
# ax.set_ylim(40,200)
ax.set_ylabel(r'Kinetic Energy')


fig.tight_layout()
fig.savefig( outputDir + out_file_name)
print "Saved Fig: ", outputDir + out_file_name


# np.savetxt('outputs_zeldovich.txt', a_list )
