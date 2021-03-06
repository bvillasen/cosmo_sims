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

eta_0 = 0.001
beta_0 = 0.055
beta_1 = 0.0


n_arg = len(sys.argv)
if n_arg > 1:
  args = []
  for i in range(1 , n_arg):
    arg = sys.argv[i]
    args.append( float( arg ))
  eta_0, beta_0, beta_1 = args
  if rank == 0:
    print "Using command arguments"
    print args

print 'eta: {0:.3f}   beta{1:.3f}  {2:.3f}/'.format( eta_0, beta_0, beta_1 )


outDir = dev_dir + 'figures/phase_diagram/uvb_eta{0:.3f}_beta{1:.3f}_{2:.3f}/'.format( eta_0, beta_0, beta_1 )


nrows = 1
chollaDir = dataDir + 'cosmo_sims/cholla_pm/256_cool/data_PPMC_HLLC_SIMPLE_eta{0:.3f}_beta{1:.3f}_{2:.3f}/'.format( eta_0, beta_0, beta_1 )


if rank == 0: 
  create_directory( outDir )
  print "Output: ", outDir

enzoDir_uv = dataDir + 'cosmo_sims/enzo/256_cool_uv/h5_files/'


gamma = 5./3
nPoints = 256
nx = nPoints
ny = nPoints
nz = nPoints
ncells = nx * ny * nz

dv = (115000./nPoints)**3
nbins = 1000

# snapshots = range(0,31)
# for nSnap in snapshots:
fileName = 'phase_diagram_{0}.png'.format(nSnap)

data_GK = []
data_U  = []
data_GE = []
data_frac = []

# for n in range( nrows ):
data_cholla = load_snapshot_data( str(nSnap), chollaDir, cool=True)
current_z_ch = data_cholla['current_z']
current_a_ch = data_cholla['current_a']
dens = data_cholla['gas']['density'][...]
dens_H = data_cholla['gas']['HI_density'][...]
dens_mean = dens.mean()

if nSnap == 0:
  px = data_cholla['gas']['momentum_x'][...]
  py = data_cholla['gas']['momentum_y'][...]
  pz = data_cholla['gas']['momentum_z'][...]
  Energy = data_cholla['gas']['Energy'][...]
  vx = px / dens
  vy = py / dens
  vz = pz / dens
  Ekin = 0.5 * dens * ( vx*vx + vy*vy + vz*vz )
  U = Energy - Ekin
  eta_vals = U / Energy
else:
  eta_vals = data_cholla['gas']['eta_value'][...]

plt.hist(eta_vals.flatten(), range=[0,1], bins= 50)
plt.savefig(outDir + 'eta_vals_{0}.png'.format(nSnap))


# temp_GK, temp_DE, temp_U, temp_GE, dens_U, dens_GE, HI_dens_U, HI_dens_GE, temp_U_ALL, temp_GE_ALL =  get_Temperaure_From_Flags_DE( data_cholla, gamma=5./3, normalize_dens=True, )
# x_GK, y_GK, z_GK = get_phase_diagram( dens.flatten()/dens_mean, temp_GK , nbins, ncells )
# x_U, y_U, z_U = get_phase_diagram( dens_U, temp_U , nbins, ncells )
# x_GE, y_GE, z_GE = get_phase_diagram( dens_GE, temp_GE , nbins, ncells )
# cell_frac = float(z_GE.sum())
# mass_frac = dens_GE.sum()*dens_mean/dens.sum()
# 
# data_GK.append( [ x_GK, y_GK, z_GK ])
# data_U.append( [ x_U, y_U, z_U ])
# data_GE.append( [ x_GE, y_GE, z_GE ])
# data_frac.append( [cell_frac, mass_frac ])



# data_enzo = load_snapshot_enzo( nSnap, enzoDir_uv, dm=False, cool=True)
# current_z_en = data_enzo['current_z']
# dens_en = data_enzo['gas']['density'][...]
# GasEnergy = data_enzo['gas']['GasEnergy'][...] / dens_en
# temp_en =  data_enzo['gas']['temperature'][...]
# dens_H_en = data_enzo['gas']['HI_density'][...]
# dens_mean = dens_en.mean()
# temp_avrg_en = temp_en.mean()
# rho_en_uv = dens_en.reshape( nx*ny*nz ) / dens_mean
# rho_H_en_uv = dens_H_en.reshape( nx*ny*nz ) / dens_mean
# temp_en_uv = temp_en.reshape( nx*ny*nz )
# dens_avrg_en_uv = dens_mean
# dens_H_avrg_en_uv = dens_H_en.mean()
# temp_avrg_en_uv = temp_avrg_en
# x_en_uv, y_en_uv, z_en_uv = get_phase_diagram( rho_en_uv, temp_en_uv , nbins, ncells )
# 
# ncols = 4
# fig, ax_l = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10*ncols,8*nrows))
# x_min = -2
# x_max = 5
# x_min_h = -11
# x_max_h = 3
# 
# y_max = 8
# 
# 
# for n in range(nrows):
# 
#   plt.subplot(nrows, ncols, n*ncols+1)
#   ax = plt.gca()
#   ax.clear()
#   c = ax.scatter( y_en_uv, x_en_uv, c = np.log10(z_en_uv), s=1,  )
#   plt.colorbar(c)
#   ax.set_ylabel(r'Log Temperature $[K]$', fontsize=15 )
#   ax.set_xlabel(r'Log Gas Overdensity', fontsize=15 )
#   ax.set_title( " Z={0:.2f}   ENZO Gas ".format( current_z_ch),fontsize=17)
#   ax.set_xlim(x_min, x_max)
#   ax.set_ylim(-1, y_max)
# 
#   data = data_GK[n]
#   plt.subplot(nrows, ncols, n*ncols+2)
#   ax = plt.gca()
#   ax.clear()
#   c = ax.scatter( data[1], data[0], c = np.log10(data[2]), s=1,  )
#   plt.colorbar(c)
#   ax.set_ylabel(r'Log Temperature $[K]$', fontsize=15 )
#   ax.set_xlabel(r'Log Gas Overdensity', fontsize=15 )
#   ax.set_title( " CHOLLA Gas GE_GRACKLE   ",fontsize=17 )
#   ax.set_xlim(x_min, x_max)
#   ax.set_ylim(-1, y_max)
# 
#   data = data_U[n]
#   plt.subplot(nrows, ncols, n*ncols+3)
#   ax = plt.gca()
#   ax.clear()
#   c = ax.scatter( data[1], data[0], c = np.log10(data[2]), s=1,  )
#   plt.colorbar(c)
#   ax.set_ylabel(r'Log Temperature $[K]$', fontsize=15 )
#   ax.set_xlabel(r'Log Gas Overdensity', fontsize=15 )
#   ax.set_title( " CHOLLA Gas GE_TOTAL   ", fontsize=17 )
#   ax.set_xlim(x_min, x_max)
#   ax.set_ylim(-1, y_max)
# 
#   data = data_GE[n]
#   plt.subplot(nrows, ncols, n*ncols+4)
#   ax = plt.gca()
#   ax.clear()
#   c = ax.scatter( data[1], data[0], c = np.log10(data[2]), s=1,  )
#   plt.colorbar(c)
#   ax.set_ylabel(r'Log Temperature $[K]$', fontsize=15 )
#   ax.set_xlabel(r'Log Gas Overdensity', fontsize=15 )
#   ax.set_title( " CHOLLA Gas GE_ADVECTED   cell_frac: {0:.3}  mass_frac: {1:.3}".format( data_frac[n][0], data_frac[n][1]), fontsize=17 )
#   ax.set_xlim(x_min, x_max)
#   ax.set_ylim(-1, y_max)
# 
# ##################
# 
# # # #
# 
# fig.suptitle(r'$\eta_0={0:0.3f}$   $\beta_0={1:0.3f}$   $\beta_1={2:0.3f}$'.format( eta_0, beta_0, beta_1 ), fontsize=20, y=0.999)
# fig.tight_layout()
# fig.savefig( outDir + fileName )
# print 'Saved image: ', fileName
# print ''
#   #
