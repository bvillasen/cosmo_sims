import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5

dev_dir = '/home/bruno/Desktop/Dropbox/Developer/'
cosmo_dir = dev_dir + 'cosmo_sims/'
toolsDirectory = cosmo_dir + "tools/"
sys.path.extend([toolsDirectory ] )
from load_data_cholla import load_snapshot_data
from load_data_ramses import load_snapshot_ramses
from phase_diagram import get_phase_diagram
from internal_energy import get_internal_energy, get_temp, get_Temperaure_From_Flags_DE
from tools import create_directory
from internal_energy import get_temp


from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nSnap = rank

# dataDir = '/home/bruno/Desktop/data/'
dataDir = '/raid/bruno/data/'



integrator = 'SIMPLE'
extra_name = ''

outDir = dev_dir + 'figures/phase_diagram/ramses/'
# outDir = dev_dir + 'figures/phase_diagram/cool_100Mpc_{0}_eta2_{1:.3f}/'.format(  integrator, eta_2 )


if rank == 0: 
  create_directory( outDir )
  print "Output: ", outDir


Lbox = 50000.
nPoints = 128

ramsesDir = dataDir + 'cosmo_sims/ramses/{0}_hydro_50Mpc_slope1/h5_files/'.format(nPoints)
chollaDir = dataDir + 'cosmo_sims/cholla_pm/{0}_hydro_50Mpc/'.format(nPoints)


nrows = 3
chollaDir_0 = chollaDir +  'data_ramses_PLMC_beta0.20_eta0.040_slope1/'
chollaDir_1 = chollaDir +  'data_ramses_PLMC_beta0.10_eta0.050_slope1/'
chollaDir_2 = chollaDir +  'data_ramses_PLMC_beta0.25_eta0.050_slope1/'
chollaDir_3 = chollaDir +  'data_ramses_eta0.500/'
chollaDir_4 = chollaDir +  'data_PPMC_HLLC_{2}_eta{0:.3f}_0.046{3}/'

chollaDir_all = [ chollaDir_0, chollaDir_1, chollaDir_2, chollaDir_3, chollaDir_4 ]
# 
beta_LIST = [  0.20, 0.10, 0.25, 0.200, 0.500, 0.050  ]
eta_LIST = [  0.040, 0.050, 0.050, 0.7, 0.200, 0.500, 0.050  ]

gamma = 5./3
nx = nPoints
ny = nPoints
nz = nPoints
ncells = nx * ny * nz

dv = (Lbox/nPoints)**3
nbins = 1000

# snapshots = range(0,31)
# for nSnap in snapshots:
fileName = 'phase_diagram_{0}.png'.format(nSnap)

data_ch = []

for n in range( nrows ):
  data_cholla = load_snapshot_data( str(nSnap), chollaDir_all[n], cool=False, single_file=False)
  current_z_ch = data_cholla['current_z']
  current_a_ch = data_cholla['current_a']
  dens_ch = data_cholla['gas']['density'][...]
  gas_U_ch = data_cholla['gas']['GasEnergy'][...] / dens_ch
  temp_ch = get_temp( gas_U_ch*1e6)
  dens_mean = dens_ch.mean()
  rho_ch = dens_ch.reshape( nx*ny*nz) / dens_mean
  temp_ch = temp_ch.reshape(nx*ny*nz)
  x_ch, y_ch, z_ch = get_phase_diagram( rho_ch, temp_ch , nbins, ncells )
  data_ch.append( [ x_ch, y_ch, z_ch ])
  
  



data_ramses = load_snapshot_ramses( nSnap, ramsesDir, dm=False, cool=False)
current_z_rm = data_ramses['current_z']
dens_rm = data_ramses['gas']['density'][...]
temp_rm =  data_ramses['gas']['temperature'][...]
dens_mean = dens_rm.mean()
temp_avrg_rm = temp_rm.mean()
rho_rm = dens_rm.reshape( nx*ny*nz ) / dens_mean
temp_rm = temp_rm.reshape( nx*ny*nz )
dens_avrg_rm = dens_mean
temp_avrg_rm = temp_avrg_rm
x_rm, y_rm, z_rm = get_phase_diagram( rho_rm, temp_rm , nbins, ncells )
# x_H_rm, y_H_rm, z_H_rm = get_phase_diagram( rho_H_rm, temp_rm , nbins, ncells )

ncols = 2
fig, ax_l = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10*ncols,8*nrows))
x_min = -2
x_max = 5
x_min_h = -11
x_max_h = 3

y_max = 8



for n in range(nrows):

  data = data_ch[n]
  min_val = min( np.min(np.log10(z_rm)), np.min(np.log10(data[2])) )
  max_val = min( np.max(np.log10(z_rm)), np.max(np.log10(data[2])) )

  plt.subplot(nrows, ncols, n*ncols+1)
  ax = plt.gca()
  ax.clear()
  c = ax.scatter( y_rm, x_rm, c = np.log10(z_rm), s=1, vmin=min_val, vmax=max_val  )
  plt.colorbar(c)
  ax.set_ylabel(r'Log Temperature $[K]$', fontsize=15 )
  ax.set_xlabel(r'Log Gas Overdensity', fontsize=15 )
  ax.set_title( " Z={0:.2f}   RAMSES Gas ".format( current_z_ch),fontsize=17)
  ax.set_xlim(x_min, x_max)
  ax.set_ylim(-3, y_max)

  plt.subplot(nrows, ncols, n*ncols+2)
  ax = plt.gca()
  ax.clear()
  c = ax.scatter( data[1], data[0], c = np.log10(data[2]), s=1, vmin=min_val, vmax=max_val )
  plt.colorbar(c)
  # plt.plot( np.log10(dens_1D), np.log10(t_jeans), c='C3', linestyle='--', linewidth=2 )
  ax.set_ylabel(r'Log Temperature $[K]$', fontsize=15 )
  ax.set_xlabel(r'Log Gas Overdensity', fontsize=15 )
  ax.set_title( r" CHOLLA Gas   $\beta={0:.2f}$  $\eta={1:.3f}$".format(beta_LIST[n], eta_LIST[n]),fontsize=17 )
  ax.set_xlim(x_min, x_max)
  ax.set_ylim(-3, y_max)

##################

# # #
# 
# fig.suptitle(r'$\eta_1={0:0.3f}$   $\eta_2={1:0.3f}$  {2}  {3} '.format( eta_1, eta_2, integrator, extra_name ), fontsize=20, y=0.999)

# fig.text( 0.44, 0.99, r'$\eta_0={0:0.3f}$   $\beta_0={1:0.3f}$   $\beta_1={2:0.3f}$'.format( eta_0, beta_0, beta_1 ), fontsize=20, )
# fig.text( 0.44, 0.49, r'$\eta_0={0:0.3f}$   $\beta_0={1:0.3f}$   $\beta_1={2:0.3f}$'.format( eta_0_1, beta_0_1, beta_1_1 ), fontsize=20, )

fig.tight_layout()
fig.savefig( outDir + fileName )
print 'Saved image: ', fileName
print ''
  #
