import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5

dev_dir = '/home/bruno/Desktop/Dropbox/Developer/'
cosmo_dir = dev_dir + 'cosmo_sims/'
toolsDirectory = cosmo_dir + "tools/"
sys.path.extend([toolsDirectory ] )
from load_data_cholla import load_snapshot_data
from load_data_enzo import load_snapshot_enzo
from phase_diagram import get_phase_diagram
from internal_energy import get_internal_energy, get_temp, get_Temperaure_From_Flags_DE
from tools import create_directory

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nSnap = rank

# name = 'de001_PLMP_CTU'
dataDir = '/raid/bruno/data/'

eta_1 = 0.001
eta_2 = 0.036



n_arg = len(sys.argv)
if n_arg > 1:
  args = []
  for i in range(1 , n_arg):
    arg = sys.argv[i]
    args.append( float( arg ))
  eta_1, eta_2 = args
  if rank == 0:
    print "Using command arguments"
    print args

print 'eta: {0:.3f}   {1:.3f}  /'.format( eta_1, eta_2 )

integrator = 'SIMPLE'
extra_name = ''

outDir = dev_dir + 'figures/phase_diagram/uvb_SIMPLE_PPMP/'
# outDir = dev_dir + 'figures/phase_diagram/cool_uv_100Mpc_{0}_eta2_{1:.3f}/'.format(  integrator, eta_2 )


if rank == 0: 
  create_directory( outDir )
  print "Output: ", outDir


Lbox = 50000.
nPoints = 128

enzoDir_uv = dataDir + 'cosmo_sims/enzo/{0}_cool_uv/h5_files/'.format(nPoints)
chollaDir = dataDir + 'cosmo_sims/cholla_pm/{0}_cool/'.format(nPoints)


nrows = 3
chollaDir_0 = chollaDir +  'data_PPMP_HLLC_SIMPLE_eta0.001_0.025/'
chollaDir_1 = chollaDir +  'data_PPMP_HLLC_SIMPLE_eta0.001_0.030/'
chollaDir_2 = chollaDir +  'data_PPMP_HLLC_SIMPLE_eta0.001_0.035/'
chollaDir_3 = chollaDir +  'data_PPMC_HLLC_{2}_eta{0:.3f}_0.038{3}/'
chollaDir_4 = chollaDir +  'data_PPMC_HLLC_{2}_eta{0:.3f}_0.046{3}/'

chollaDir_all = [ chollaDir_0, chollaDir_1, chollaDir_2, chollaDir_3, chollaDir_4 ]
# 
eta_2_LIST = [ 0.025, 0.030, 0.035,  0.040, 0.045, 0.050  ]
# nrows = len( eta_2_LIST )
# chollaDir_all = []
# for eta_2 in eta_2_LIST:
#   chollaDir_0 = chollaDir +  'data_PPMC_HLLC_{2}_eta{0:.3f}_{1:.4f}{3}/'.format( eta_1, eta_2, integrator, extra_name )
#   chollaDir_all.append( chollaDir_0)
# 
# 


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
  data_cholla = load_snapshot_data( str(nSnap), chollaDir_all[n], cool=True)
  current_z_ch = data_cholla['current_z']
  current_a_ch = data_cholla['current_a']
  dens_ch = data_cholla['gas']['density'][...]
  dens_H_ch = data_cholla['gas']['HI_density'][...]
  temp_ch =  data_cholla['gas']['temperature'][...]
  dens_mean = dens_ch.mean()
  rho_ch = dens_ch.reshape( nx*ny*nz) / dens_mean
  temp_ch = temp_ch.reshape(nx*ny*nz)
  x_ch_uv, y_ch_uv, z_ch_uv = get_phase_diagram( rho_ch, temp_ch , nbins, ncells )
  data_ch.append( [ x_ch_uv, y_ch_uv, z_ch_uv ])
  
  



data_enzo = load_snapshot_enzo( nSnap, enzoDir_uv, dm=False, cool=True)
current_z_en = data_enzo['current_z']
dens_en = data_enzo['gas']['density'][...]
GasEnergy = data_enzo['gas']['GasEnergy'][...] / dens_en
temp_en =  data_enzo['gas']['temperature'][...]
dens_H_en = data_enzo['gas']['HI_density'][...]
dens_mean = dens_en.mean()
temp_avrg_en = temp_en.mean()
rho_en_uv = dens_en.reshape( nx*ny*nz ) / dens_mean
rho_H_en_uv = dens_H_en.reshape( nx*ny*nz ) / dens_mean
temp_en_uv = temp_en.reshape( nx*ny*nz )
dens_avrg_en_uv = dens_mean
dens_H_avrg_en_uv = dens_H_en.mean()
temp_avrg_en_uv = temp_avrg_en
x_en_uv, y_en_uv, z_en_uv = get_phase_diagram( rho_en_uv, temp_en_uv , nbins, ncells )
# x_H_en_uv, y_H_en_uv, z_H_en_uv = get_phase_diagram( rho_H_en_uv, temp_en_uv , nbins, ncells )

ncols = 2
fig, ax_l = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10*ncols,8*nrows))
x_min = -2
x_max = 5
x_min_h = -11
x_max_h = 3

y_max = 8



for n in range(nrows):

  data = data_ch[n]
  min_val = min( np.min(np.log10(z_en_uv)), np.min(np.log10(data[2])) )
  max_val = min( np.max(np.log10(z_en_uv)), np.max(np.log10(data[2])) )

  plt.subplot(nrows, ncols, n*ncols+1)
  ax = plt.gca()
  ax.clear()
  c = ax.scatter( y_en_uv, x_en_uv, c = np.log10(z_en_uv), s=1, vmin=min_val, vmax=max_val  )
  plt.colorbar(c)
  ax.set_ylabel(r'Log Temperature $[K]$', fontsize=15 )
  ax.set_xlabel(r'Log Gas Overdensity', fontsize=15 )
  ax.set_title( " Z={0:.2f}   ENZO Gas ".format( current_z_ch),fontsize=17)
  ax.set_xlim(x_min, x_max)
  ax.set_ylim(-1, y_max)

  plt.subplot(nrows, ncols, n*ncols+2)
  ax = plt.gca()
  ax.clear()
  c = ax.scatter( data[1], data[0], c = np.log10(data[2]), s=1, vmin=min_val, vmax=max_val )
  plt.colorbar(c)
  # plt.plot( np.log10(dens_1D), np.log10(t_jeans), c='C3', linestyle='--', linewidth=2 )
  ax.set_ylabel(r'Log Temperature $[K]$', fontsize=15 )
  ax.set_xlabel(r'Log Gas Overdensity', fontsize=15 )
  ax.set_title( r" CHOLLA Gas   $\eta_2={0:.3f}$".format(eta_2_LIST[n]),fontsize=17 )
  ax.set_xlim(x_min, x_max)
  ax.set_ylim(-1, y_max)

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
