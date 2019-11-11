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


data_name = 'SIMPLE_PPMP_eta0.030_beta0.00_grav4'

outDir = dev_dir + 'figures/phase_diagram/enzo_{0}/'.format( data_name )
# outDir = dev_dir + 'figures/phase_diagram/cool_uv_100Mpc_{0}_eta2_{1:.3f}/'.format(  integrator, eta_2 )


if rank == 0: 
  create_directory( outDir )
  print "Output: ", outDir


Lbox = 50000.
nPoints = 256

enzoDir_uv = dataDir + 'cosmo_sims/enzo/{0}_cool_uv_50Mpc_HLLC_grav4/h5_files/'.format(nPoints)
chollaDir = dataDir + 'cosmo_sims/cholla_pm/{0}_cool_uv_50Mpc/'.format(nPoints)

nrows = 1
# 
chollaDir_0 = chollaDir +  'data_{0}/'.format( data_name )
 



gamma = 5./3
nx = nPoints
ny = nPoints
nz = nPoints
ncells = nx * ny * nz

dv = (Lbox/nPoints)**3
nbins = 1000

nSnap = rank

# snapshots = range(0,31)
# for nSnap in snapshots:
fileName = 'phase_diagram_{0}.png'.format(nSnap)

data_GK = []
data_H_GK = []
data_U  = []
data_GE = []
data_frac = []

data_cholla = load_snapshot_data( str(nSnap), chollaDir_0, cool=True)
current_z_ch = data_cholla['current_z']
current_a_ch = data_cholla['current_a']
dens = data_cholla['gas']['density'][...]
dens_H = data_cholla['gas']['HI_density'][...]
dens_mean = dens.mean()
temp_GK = data_cholla['gas']['temperature'][...].flatten()
# temp_GK, temp_DE, temp_U, temp_GE, dens_U, dens_GE, HI_dens_U, HI_dens_GE, temp_U_ALL, temp_GE_ALL =  get_Temperaure_From_Flags_DE( data_cholla, gamma=5./3, normalize_dens=True, )
# x_U_ALL_0, y_U_ALL_0, z_U_ALL_0 = get_phase_diagram( dens.flatten()/dens_mean, temp_U_ALL , nbins, ncells )
# x_GE_ALL_0, y_GE_ALL_0, z_GE_ALL_0 = get_phase_diagram( dens.flatten()/dens_mean, temp_GE_ALL , nbins, ncells )
x_GK, y_GK, z_GK = get_phase_diagram( dens.flatten()/dens_mean, temp_GK , nbins, ncells )
x_H_GK, y_H_GK, z_H_GK = get_phase_diagram( dens_H.flatten()/dens_mean, temp_GK , nbins, ncells )
# x_U, y_U, z_U = get_phase_diagram( dens_U, temp_U , nbins, ncells )
# x_GE, y_GE, z_GE = get_phase_diagram( dens_GE, temp_GE , nbins, ncells )
# cell_frac = float(z_GE.sum())
# mass_frac = dens_GE.sum()*dens_mean/dens.sum()
# 
data_GK.append( [ x_GK, y_GK, z_GK ])
data_H_GK.append( [ x_H_GK, y_H_GK, z_H_GK ])
# data_U.append( [ x_U, y_U, z_U ])
# data_GE.append( [ x_GE, y_GE, z_GE ])
# data_frac.append( [cell_frac, mass_frac ])
# 
# 
# dens_1D = np.linspace( 0.001, 1e5, 1000  )
# alpha = 500
# p_jeans = alpha * dens_1D**2
# t_jeans = p_jeans / dens_1D 
# 
# 

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
x_H_en_uv, y_H_en_uv, z_H_en_uv = get_phase_diagram( rho_H_en_uv, temp_en_uv , nbins, ncells )

nrows=2
ncols = 2
fig, ax_l = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10*ncols,8*nrows))
x_min = -2
x_max = 4
x_min_h = -11
x_max_h = 3

y_max = 8
y_min = 0


fs = 17
fs_1 = 20

n = 0

data = data_GK[n]
data_H = data_H_GK[n]

min_val = min( np.min(np.log10(z_en_uv)), np.min(np.log10(data[2])) )
max_val = min( np.max(np.log10(z_en_uv)), np.max(np.log10(data[2])) )

min_val_H = min( np.min(np.log10(z_H_en_uv)), np.min(np.log10(data_H[2])) )
max_val_H = min( np.max(np.log10(z_H_en_uv)), np.max(np.log10(data_H[2])) )

plt.subplot(nrows, ncols, n*ncols+1)
ax = plt.gca()
ax.clear()
c = ax.scatter( y_en_uv, x_en_uv, c = np.log10(z_en_uv), s=1, vmin=min_val, vmax=max_val  )
plt.colorbar(c)
ax.set_ylabel(r'Log Temperature $[K]$', fontsize=fs )
ax.set_xlabel(r'Log Gas Overdensity', fontsize=fs )
ax.set_title( " Z={0:.2f}   ENZO Gas ".format( np.abs(current_z_ch)),fontsize=fs_1)
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

plt.subplot(nrows, ncols, n*ncols+2)
ax = plt.gca()
ax.clear()
c = ax.scatter( data[1], data[0], c = np.log10(data[2]), s=1, vmin=min_val, vmax=max_val )
plt.colorbar(c)
# plt.plot( np.log10(dens_1D), np.log10(t_jeans), c='C3', linestyle='--', linewidth=2 )
ax.set_ylabel(r'Log Temperature $[K]$', fontsize=fs )
ax.set_xlabel(r'Log Gas Overdensity', fontsize=fs )
ax.set_title( r" CHOLLA Gas   ",fontsize=fs_1 )
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

plt.subplot(nrows, ncols, ncols+1)
ax = plt.gca()
ax.clear()
c = ax.scatter( y_H_en_uv, x_H_en_uv, c = np.log10(z_H_en_uv), s=1, vmin=min_val_H, vmax=max_val_H  )
plt.colorbar(c)
ax.set_ylabel(r'Log Temperature $[K]$', fontsize=fs )
ax.set_xlabel(r'Log HI Overdensity', fontsize=fs )
ax.set_title( "ENZO Neutral Hydrogen ".format( current_z_ch),fontsize=fs_1)
ax.set_xlim(-12, 2)
ax.set_ylim(y_min, y_max)



plt.subplot(nrows, ncols, ncols+2)
ax = plt.gca()
ax.clear()
c = ax.scatter( data_H[1], data_H[0], c = np.log10(data_H[2]), s=1, vmin=min_val_H, vmax=max_val_H )
plt.colorbar(c)
# plt.plot( np.log10(dens_1D), np.log10(t_jeans), c='C3', linestyle='--', linewidth=2 )
ax.set_ylabel(r'Log Temperature $[K]$', fontsize=fs )
ax.set_xlabel(r'Log HI Overdensity', fontsize=fs )
ax.set_title( r" CHOLLA Neutral Hydrogen   ",fontsize=fs_1 )
ax.set_xlim(-12, 2)
ax.set_ylim(y_min, y_max)

##################

# # #
# 
fig.suptitle('{0}'.format( data_name ), fontsize=20, y=0.999)

# fig.text( 0.44, 0.99, r'$\eta_0={0:0.3f}$   $\beta_0={1:0.3f}$   $\beta_1={2:0.3f}$'.format( eta_0, beta_0, beta_1 ), fontsize=20, )
# fig.text( 0.44, 0.49, r'$\eta_0={0:0.3f}$   $\beta_0={1:0.3f}$   $\beta_1={2:0.3f}$'.format( eta_0_1, beta_0_1, beta_1_1 ), fontsize=20, )

fig.tight_layout()
fig.savefig( outDir + fileName )
print 'Saved image: ', fileName
print ''
  #
