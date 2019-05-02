import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5

cosmo_dir = '/home/bruno/Desktop/Dropbox/Developer/cosmo_sims/'
outDir = cosmo_dir + 'figures/phase_diagram/'
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

dataDir = '/raid/bruno/data/'
outDir = cosmo_dir + 'figures/phase_diagram/uvb_de02_noGrav/'


nrows = 1
chollaDir_0 = dataDir + 'cosmo_sims/cholla_pm/256_cool/data_de02_noGrav/'
chollaDir_1 = dataDir + 'cosmo_sims/cholla_pm/256_cool/data_de02_PPMC_da00001/'
chollaDir_2 = dataDir + 'cosmo_sims/cholla_pm/256_cool/data_de02_PLMC_noFirst/'
chollaDir_3 = dataDir + 'cosmo_sims/cholla_pm/256_cool/data_de02_PPMC_noFirst/'
# chollaDir_4 = dataDir + 'cosmo_sims/cholla_pm/256_cool/data_de02_PPMC/'


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

snapshots = range(0,31)
# for nSnap in snapshots:
fileName = 'phase_diagram_{0}.png'.format(nSnap)

data_cholla = load_snapshot_data( str(nSnap), chollaDir_0, cool=True)
current_z_ch = data_cholla['current_z']
current_a_ch = data_cholla['current_a']
dens = data_cholla['gas']['density'][...]
dens_H = data_cholla['gas']['HI_density'][...]
dens_mean = dens.mean()
temp_GK, temp_DE, temp_U, temp_GE, dens_U, dens_GE, HI_dens_U, HI_dens_GE, temp_U_ALL, temp_GE_ALL =  get_Temperaure_From_Flags_DE( data_cholla, gamma=5./3, normalize_dens=True, )
x_U_ALL_0, y_U_ALL_0, z_U_ALL_0 = get_phase_diagram( dens.flatten()/dens_mean, temp_U_ALL , nbins, ncells )
x_GE_ALL_0, y_GE_ALL_0, z_GE_ALL_0 = get_phase_diagram( dens.flatten()/dens_mean, temp_GE_ALL , nbins, ncells )
x_GK_0, y_GK_0, z_GK_0 = get_phase_diagram( dens.flatten()/dens_mean, temp_GK , nbins, ncells )
x_U_0, y_U_0, z_U_0 = get_phase_diagram( dens_U, temp_U , nbins, ncells )
x_GE_0, y_GE_0, z_GE_0 = get_phase_diagram( dens_GE, temp_GE , nbins, ncells )
cell_frac_0 = float(z_GE_0.sum())
mass_frac_0 = dens_GE.sum()*dens_mean/dens.sum()

if nrows > 1:
  data_cholla = load_snapshot_data( str(nSnap), chollaDir_1, cool=True)
  current_z_ch = data_cholla['current_z']
  current_a_ch = data_cholla['current_a']
  dens = data_cholla['gas']['density'][...]
  dens_H = data_cholla['gas']['HI_density'][...]
  dens_mean = dens.mean()
  temp_GK, temp_DE, temp_U, temp_GE, dens_U, dens_GE, HI_dens_U, HI_dens_GE, temp_U_ALL, temp_GE_ALL =  get_Temperaure_From_Flags_DE( data_cholla, gamma=5./3, normalize_dens=True, )
  x_U_ALL_1, y_U_ALL_1, z_U_ALL_1 = get_phase_diagram( dens.flatten()/dens_mean, temp_U_ALL , nbins, ncells )
  x_GE_ALL_1, y_GE_ALL_1, z_GE_ALL_1 = get_phase_diagram( dens.flatten()/dens_mean, temp_GE_ALL , nbins, ncells )
  x_GK_1, y_GK_1, z_GK_1 = get_phase_diagram( dens.flatten()/dens_mean, temp_GK , nbins, ncells )
  x_U_1, y_U_1, z_U_1 = get_phase_diagram( dens_U, temp_U , nbins, ncells )
  x_GE_1, y_GE_1, z_GE_1 = get_phase_diagram( dens_GE, temp_GE , nbins, ncells )
  cell_frac_1 = float(z_GE_1.sum())
  mass_frac_1 = dens_GE.sum()*dens_mean/dens.sum()

if nrows > 2:
  data_cholla = load_snapshot_data( str(nSnap), chollaDir_2, cool=True)
  current_z_ch = data_cholla['current_z']
  current_a_ch = data_cholla['current_a']
  dens = data_cholla['gas']['density'][...]
  dens_H = data_cholla['gas']['HI_density'][...]
  dens_mean = dens.mean()
  temp_GK, temp_DE, temp_U, temp_GE, dens_U, dens_GE, HI_dens_U, HI_dens_GE, temp_U_ALL, temp_GE_ALL =  get_Temperaure_From_Flags_DE( data_cholla, gamma=5./3, normalize_dens=True, )
  x_U_ALL_2, y_U_ALL_2, z_U_ALL_2 = get_phase_diagram( dens.flatten()/dens_mean, temp_U_ALL , nbins, ncells )
  x_GE_ALL_2, y_GE_ALL_2, z_GE_ALL_2 = get_phase_diagram( dens.flatten()/dens_mean, temp_GE_ALL , nbins, ncells )
  x_GK_2, y_GK_2, z_GK_2 = get_phase_diagram( dens.flatten()/dens_mean, temp_GK , nbins, ncells )
  x_U_2, y_U_2, z_U_2 = get_phase_diagram( dens_U, temp_U , nbins, ncells )
  x_GE_2, y_GE_2, z_GE_2 = get_phase_diagram( dens_GE, temp_GE , nbins, ncells )
  cell_frac_2 = float(z_GE_2.sum())
  mass_frac_2 = dens_GE.sum()*dens_mean/dens.sum()

if nrows > 3:
  data_cholla = load_snapshot_data( str(nSnap), chollaDir_3, cool=True)
  current_z_ch = data_cholla['current_z']
  current_a_ch = data_cholla['current_a']
  dens = data_cholla['gas']['density'][...]
  dens_H = data_cholla['gas']['HI_density'][...]
  dens_mean = dens.mean()
  temp_GK, temp_DE, temp_U, temp_GE, dens_U, dens_GE, HI_dens_U, HI_dens_GE, temp_U_ALL, temp_GE_ALL =  get_Temperaure_From_Flags_DE( data_cholla, gamma=5./3, normalize_dens=True, )
  x_U_ALL_3, y_U_ALL_3, z_U_ALL_3 = get_phase_diagram( dens.flatten()/dens_mean, temp_U_ALL , nbins, ncells )
  x_GE_ALL_3, y_GE_ALL_3, z_GE_ALL_3 = get_phase_diagram( dens.flatten()/dens_mean, temp_GE_ALL , nbins, ncells )
  x_GK_3, y_GK_3, z_GK_3 = get_phase_diagram( dens.flatten()/dens_mean, temp_GK , nbins, ncells )
  x_U_3, y_U_3, z_U_3 = get_phase_diagram( dens_U, temp_U , nbins, ncells )
  x_GE_3, y_GE_3, z_GE_3 = get_phase_diagram( dens_GE, temp_GE , nbins, ncells )
  cell_frac_3 = float(z_GE_3.sum())
  mass_frac_3 = dens_GE.sum()*dens_mean/dens.sum()

if nrows > 4:
  data_cholla = load_snapshot_data( str(nSnap), chollaDir_4, cool=True)
  current_z_ch = data_cholla['current_z']
  current_a_ch = data_cholla['current_a']
  dens = data_cholla['gas']['density'][...]
  dens_H = data_cholla['gas']['HI_density'][...]
  dens_mean = dens.mean()
  temp_GK, temp_DE, temp_U, temp_GE, dens_U, dens_GE, HI_dens_U, HI_dens_GE, temp_U_ALL, temp_GE_ALL =  get_Temperaure_From_Flags_DE( data_cholla, gamma=5./3, normalize_dens=True, )
  x_U_ALL_4, y_U_ALL_4, z_U_ALL_4 = get_phase_diagram( dens.flatten()/dens_mean, temp_U_ALL , nbins, ncells )
  x_GE_ALL_4, y_GE_ALL_4, z_GE_ALL_4 = get_phase_diagram( dens.flatten()/dens_mean, temp_GE_ALL , nbins, ncells )
  x_GK_4, y_GK_4, z_GK_4 = get_phase_diagram( dens.flatten()/dens_mean, temp_GK , nbins, ncells )
  x_U_4, y_U_4, z_U_4 = get_phase_diagram( dens_U, temp_U , nbins, ncells )
  x_GE_4, y_GE_4, z_GE_4 = get_phase_diagram( dens_GE, temp_GE , nbins, ncells )
  cell_frac_4 = float(z_GE_4.sum())
  mass_frac_4 = dens_GE.sum()*dens_mean/dens.sum()


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

ncols = 6
fig, ax_l = plt.subplots(nrows=2, ncols=ncols, figsize=(10*ncols,8*nrows))
x_min = -2
x_max = 5
x_min_h = -11
x_max_h = 3

y_max = 8


plt.subplot(nrows, ncols, 1)
ax = plt.gca()
ax.clear()
c = ax.scatter( y_en_uv, x_en_uv, c = np.log10(z_en_uv), s=1,  )
plt.colorbar(c)
ax.set_ylabel(r'Log Temperature $[K]$', fontsize=15 )
ax.set_xlabel(r'Log Gas Overdensity', fontsize=15 )
ax.set_title( " Z={2:.2f}   ENZO Gas UVB HM2012  T={0:.2e}   d={1:.2e}".format( temp_avrg_en_uv, dens_avrg_en_uv, current_z_ch),fontsize=17)
ax.set_xlim(x_min, x_max)
ax.set_ylim(-1, y_max)


plt.subplot(nrows, ncols, 2)
ax = plt.gca()
ax.clear()
c = ax.scatter( y_GK_0, x_GK_0, c = np.log10(z_GK_0), s=1,  )
plt.colorbar(c)
ax.set_ylabel(r'Log Temperature $[K]$', fontsize=15 )
ax.set_xlabel(r'Log Gas Overdensity', fontsize=15 )
ax.set_title( " CHOLLA Gas GE_GRACKLE   ",fontsize=17 )
ax.set_xlim(x_min, x_max)
ax.set_ylim(-1, y_max)


plt.subplot(nrows, ncols, 3)
ax = plt.gca()
ax.clear()
c = ax.scatter( y_U_0, x_U_0, c = np.log10(z_U_0), s=1,  )
plt.colorbar(c)
ax.set_ylabel(r'Log Temperature $[K]$', fontsize=15 )
ax.set_xlabel(r'Log Gas Overdensity', fontsize=15 )
ax.set_title( " CHOLLA Gas GE_TOTAL   ", fontsize=17 )
ax.set_xlim(x_min, x_max)
ax.set_ylim(-1, y_max)

plt.subplot(nrows, ncols, 4)
ax = plt.gca()
ax.clear()
c = ax.scatter( y_GE_0, x_GE_0, c = np.log10(z_GE_0), s=1,  )
plt.colorbar(c)
ax.set_ylabel(r'Log Temperature $[K]$', fontsize=15 )
ax.set_xlabel(r'Log Gas Overdensity', fontsize=15 )
ax.set_title( " CHOLLA Gas GE_ADVECTED   cell_frac: {0:.3}  mass_frac: {1:.3}".format( cell_frac_0, mass_frac_0), fontsize=17 )
ax.set_xlim(x_min, x_max)
ax.set_ylim(-1, y_max)


plt.subplot(nrows, ncols, 5)
ax = plt.gca()
ax.clear()
c = ax.scatter( y_U_ALL_0, x_U_ALL_0, c = np.log10(z_U_ALL_0), s=1,  )
plt.colorbar(c)
ax.set_ylabel(r'Log Temperature $[K]$', fontsize=15 )
ax.set_xlabel(r'Log Gas Overdensity', fontsize=15 )
ax.set_title( " CHOLLA Gas GE_TOTAL  ALL ", fontsize=17 )
ax.set_xlim(x_min, x_max)
ax.set_ylim(-1, y_max)

plt.subplot(nrows, ncols, 6)
ax = plt.gca()
ax.clear()
c = ax.scatter( y_GE_ALL_0, x_GE_ALL_0, c = np.log10(z_GE_ALL_0), s=1,  )
plt.colorbar(c)
ax.set_ylabel(r'Log Temperature $[K]$', fontsize=15 )
ax.set_xlabel(r'Log Gas Overdensity', fontsize=15 )
ax.set_title( " CHOLLA Gas GE_ADVECTED ALL  ", fontsize=17 )
ax.set_xlim(x_min, x_max)
ax.set_ylim(-1, y_max)

#################################################################################################################

if nrows > 1:
  plt.subplot(nrows, ncols, ncols+1)
  ax = plt.gca()
  ax.clear()
  c = ax.scatter( y_en_uv, x_en_uv, c = np.log10(z_en_uv), s=1,  )
  plt.colorbar(c)
  ax.set_ylabel(r'Log Temperature $[K]$', fontsize=15 )
  ax.set_xlabel(r'Log Gas Overdensity', fontsize=15 )
  ax.set_title( " Z={2:.2f}   ENZO Gas UVB HM2012  T={0:.2e}   d={1:.2e}".format( temp_avrg_en_uv, dens_avrg_en_uv, current_z_ch),fontsize=17)
  ax.set_xlim(x_min, x_max)
  ax.set_ylim(-1, y_max)


  plt.subplot(nrows, ncols, ncols+2)
  ax = plt.gca()
  ax.clear()
  c = ax.scatter( y_GK_1, x_GK_1, c = np.log10(z_GK_1), s=1,  )
  plt.colorbar(c)
  ax.set_ylabel(r'Log Temperature $[K]$', fontsize=15 )
  ax.set_xlabel(r'Log Gas Overdensity', fontsize=15 )
  ax.set_title( " CHOLLA Gas GE_GRACKLE   ",fontsize=17 )
  ax.set_xlim(x_min, x_max)
  ax.set_ylim(-1, y_max)


  plt.subplot(nrows, ncols, ncols+3)
  ax = plt.gca()
  ax.clear()
  c = ax.scatter( y_U_1, x_U_1, c = np.log10(z_U_1), s=1,  )
  plt.colorbar(c)
  ax.set_ylabel(r'Log Temperature $[K]$', fontsize=15 )
  ax.set_xlabel(r'Log Gas Overdensity', fontsize=15 )
  ax.set_title( " CHOLLA Gas GE_TOTAL   ", fontsize=17 )
  ax.set_xlim(x_min, x_max)
  ax.set_ylim(-1, y_max)

  plt.subplot(nrows, ncols, ncols+4)
  ax = plt.gca()
  ax.clear()
  c = ax.scatter( y_GE_1, x_GE_1, c = np.log10(z_GE_1), s=1,  )
  plt.colorbar(c)
  ax.set_ylabel(r'Log Temperature $[K]$', fontsize=15 )
  ax.set_xlabel(r'Log Gas Overdensity', fontsize=15 )
  ax.set_title( " CHOLLA Gas GE_ADVECTED   cell_frac: {0:.3}  mass_frac: {1:.3}".format( cell_frac_1, mass_frac_1), fontsize=17 )
  ax.set_xlim(x_min, x_max)
  ax.set_ylim(-1, y_max)


  plt.subplot(nrows, ncols, ncols+5)
  ax = plt.gca()
  ax.clear()
  c = ax.scatter( y_U_ALL_1, x_U_ALL_1, c = np.log10(z_U_ALL_1), s=1,  )
  plt.colorbar(c)
  ax.set_ylabel(r'Log Temperature $[K]$', fontsize=15 )
  ax.set_xlabel(r'Log Gas Overdensity', fontsize=15 )
  ax.set_title( " CHOLLA Gas GE_TOTAL  ALL ", fontsize=17 )
  ax.set_xlim(x_min, x_max)
  ax.set_ylim(-1, y_max)

  plt.subplot(nrows, ncols, ncols+6)
  ax = plt.gca()
  ax.clear()
  c = ax.scatter( y_GE_ALL_1, x_GE_ALL_1, c = np.log10(z_GE_ALL_1), s=1,  )
  plt.colorbar(c)
  ax.set_ylabel(r'Log Temperature $[K]$', fontsize=15 )
  ax.set_xlabel(r'Log Gas Overdensity', fontsize=15 )
  ax.set_title( " CHOLLA Gas GE_ADVECTED ALL  ", fontsize=17 )
  ax.set_xlim(x_min, x_max)
  ax.set_ylim(-1, y_max)

#################################################################################################################

if nrows > 2:
  plt.subplot(nrows, ncols, 2*ncols+1)
  ax = plt.gca()
  ax.clear()
  c = ax.scatter( y_en_uv, x_en_uv, c = np.log10(z_en_uv), s=1,  )
  plt.colorbar(c)
  ax.set_ylabel(r'Log Temperature $[K]$', fontsize=15 )
  ax.set_xlabel(r'Log Gas Overdensity', fontsize=15 )
  ax.set_title( " Z={2:.2f}   ENZO Gas UVB HM2012  T={0:.2e}   d={1:.2e}".format( temp_avrg_en_uv, dens_avrg_en_uv, current_z_ch),fontsize=17)
  ax.set_xlim(x_min, x_max)
  ax.set_ylim(-1, y_max)


  plt.subplot(nrows, ncols, 2*ncols+2)
  ax = plt.gca()
  ax.clear()
  c = ax.scatter( y_GK_2, x_GK_2, c = np.log10(z_GK_2), s=1,  )
  plt.colorbar(c)
  ax.set_ylabel(r'Log Temperature $[K]$', fontsize=15 )
  ax.set_xlabel(r'Log Gas Overdensity', fontsize=15 )
  ax.set_title( " CHOLLA Gas GE_GRACKLE   ",fontsize=17 )
  ax.set_xlim(x_min, x_max)
  ax.set_ylim(-1, y_max)


  plt.subplot(nrows, ncols, 2*ncols+3)
  ax = plt.gca()
  ax.clear()
  c = ax.scatter( y_U_2, x_U_2, c = np.log10(z_U_2), s=1,  )
  plt.colorbar(c)
  ax.set_ylabel(r'Log Temperature $[K]$', fontsize=15 )
  ax.set_xlabel(r'Log Gas Overdensity', fontsize=15 )
  ax.set_title( " CHOLLA Gas GE_TOTAL   ", fontsize=17 )
  ax.set_xlim(x_min, x_max)
  ax.set_ylim(-1, y_max)

  plt.subplot(nrows, ncols, 2*ncols+4)
  ax = plt.gca()
  ax.clear()
  c = ax.scatter( y_GE_2, x_GE_2, c = np.log10(z_GE_2), s=1,  )
  plt.colorbar(c)
  ax.set_ylabel(r'Log Temperature $[K]$', fontsize=15 )
  ax.set_xlabel(r'Log Gas Overdensity', fontsize=15 )
  ax.set_title( " CHOLLA Gas GE_ADVECTED   cell_frac: {0:.3}  mass_frac: {1:.3}".format( cell_frac_2, mass_frac_2), fontsize=17 )
  ax.set_xlim(x_min, x_max)
  ax.set_ylim(-1, y_max)


  plt.subplot(nrows, ncols, 2*ncols+5)
  ax = plt.gca()
  ax.clear()
  c = ax.scatter( y_U_ALL_2, x_U_ALL_2, c = np.log10(z_U_ALL_2), s=1,  )
  plt.colorbar(c)
  ax.set_ylabel(r'Log Temperature $[K]$', fontsize=15 )
  ax.set_xlabel(r'Log Gas Overdensity', fontsize=15 )
  ax.set_title( " CHOLLA Gas GE_TOTAL  ALL ", fontsize=17 )
  ax.set_xlim(x_min, x_max)
  ax.set_ylim(-1, y_max)

  plt.subplot(nrows, ncols, 2*ncols+6)
  ax = plt.gca()
  ax.clear()
  c = ax.scatter( y_GE_ALL_2, x_GE_ALL_2, c = np.log10(z_GE_ALL_2), s=1,  )
  plt.colorbar(c)
  ax.set_ylabel(r'Log Temperature $[K]$', fontsize=15 )
  ax.set_xlabel(r'Log Gas Overdensity', fontsize=15 )
  ax.set_title( " CHOLLA Gas GE_ADVECTED ALL  ", fontsize=17 )
  ax.set_xlim(x_min, x_max)
  ax.set_ylim(-1, y_max)

# #################################################################################################################

if nrows > 3:
  plt.subplot(nrows, ncols, 3*ncols+1)
  ax = plt.gca()
  ax.clear()
  c = ax.scatter( y_en_uv, x_en_uv, c = np.log10(z_en_uv), s=1,  )
  plt.colorbar(c)
  ax.set_ylabel(r'Log Temperature $[K]$', fontsize=15 )
  ax.set_xlabel(r'Log Gas Overdensity', fontsize=15 )
  ax.set_title( " Z={2:.2f}   ENZO Gas UVB HM2012  T={0:.2e}   d={1:.2e}".format( temp_avrg_en_uv, dens_avrg_en_uv, current_z_ch),fontsize=17)
  ax.set_xlim(x_min, x_max)
  ax.set_ylim(-1, y_max)


  plt.subplot(nrows, ncols, 3*ncols+2)
  ax = plt.gca()
  ax.clear()
  c = ax.scatter( y_GK_3, x_GK_3, c = np.log10(z_GK_3), s=1,  )
  plt.colorbar(c)
  ax.set_ylabel(r'Log Temperature $[K]$', fontsize=15 )
  ax.set_xlabel(r'Log Gas Overdensity', fontsize=15 )
  ax.set_title( " CHOLLA Gas GE_GRACKLE   ",fontsize=17 )
  ax.set_xlim(x_min, x_max)
  ax.set_ylim(-1, y_max)


  plt.subplot(nrows, ncols, 3*ncols+3)
  ax = plt.gca()
  ax.clear()
  c = ax.scatter( y_U_3, x_U_3, c = np.log10(z_U_3), s=1,  )
  plt.colorbar(c)
  ax.set_ylabel(r'Log Temperature $[K]$', fontsize=15 )
  ax.set_xlabel(r'Log Gas Overdensity', fontsize=15 )
  ax.set_title( " CHOLLA Gas GE_TOTAL   ", fontsize=17 )
  ax.set_xlim(x_min, x_max)
  ax.set_ylim(-1, y_max)

  plt.subplot(nrows, ncols, 3*ncols+4)
  ax = plt.gca()
  ax.clear()
  c = ax.scatter( y_GE_3, x_GE_3, c = np.log10(z_GE_3), s=1,  )
  plt.colorbar(c)
  ax.set_ylabel(r'Log Temperature $[K]$', fontsize=15 )
  ax.set_xlabel(r'Log Gas Overdensity', fontsize=15 )
  ax.set_title( " CHOLLA Gas GE_ADVECTED   cell_frac: {0:.3}  mass_frac: {1:.3}".format( cell_frac_3, mass_frac_3), fontsize=17 )
  ax.set_xlim(x_min, x_max)
  ax.set_ylim(-1, y_max)


  plt.subplot(nrows, ncols, 3*ncols+5)
  ax = plt.gca()
  ax.clear()
  c = ax.scatter( y_U_ALL_3, x_U_ALL_3, c = np.log10(z_U_ALL_3), s=1,  )
  plt.colorbar(c)
  ax.set_ylabel(r'Log Temperature $[K]$', fontsize=15 )
  ax.set_xlabel(r'Log Gas Overdensity', fontsize=15 )
  ax.set_title( " CHOLLA Gas GE_TOTAL  ALL ", fontsize=17 )
  ax.set_xlim(x_min, x_max)
  ax.set_ylim(-1, y_max)

  plt.subplot(nrows, ncols, 3*ncols+6)
  ax = plt.gca()
  ax.clear()
  c = ax.scatter( y_GE_ALL_3, x_GE_ALL_3, c = np.log10(z_GE_ALL_3), s=1,  )
  plt.colorbar(c)
  ax.set_ylabel(r'Log Temperature $[K]$', fontsize=15 )
  ax.set_xlabel(r'Log Gas Overdensity', fontsize=15 )
  ax.set_title( " CHOLLA Gas GE_ADVECTED ALL  ", fontsize=17 )
  ax.set_xlim(x_min, x_max)
  ax.set_ylim(-1, y_max)

  ###########################
if nrows > 4:
  plt.subplot(nrows, ncols, 4*ncols+1)
  ax = plt.gca()
  ax.clear()
  c = ax.scatter( y_en_uv, x_en_uv, c = np.log10(z_en_uv), s=1,  )
  plt.colorbar(c)
  ax.set_ylabel(r'Log Temperature $[K]$', fontsize=15 )
  ax.set_xlabel(r'Log Gas Overdensity', fontsize=15 )
  ax.set_title( " Z={2:.2f}   ENZO Gas UVB HM2012  T={0:.2e}   d={1:.2e}".format( temp_avrg_en_uv, dens_avrg_en_uv, current_z_ch),fontsize=17)
  ax.set_xlim(x_min, x_max)
  ax.set_ylim(-1, y_max)


  plt.subplot(nrows, ncols, 4*ncols+2)
  ax = plt.gca()
  ax.clear()
  c = ax.scatter( y_GK_4, x_GK_4, c = np.log10(z_GK_4), s=1,  )
  plt.colorbar(c)
  ax.set_ylabel(r'Log Temperature $[K]$', fontsize=15 )
  ax.set_xlabel(r'Log Gas Overdensity', fontsize=15 )
  ax.set_title( " CHOLLA Gas GE_GRACKLE   ",fontsize=17 )
  ax.set_xlim(x_min, x_max)
  ax.set_ylim(-1, y_max)


  plt.subplot(nrows, ncols, 4*ncols+3)
  ax = plt.gca()
  ax.clear()
  c = ax.scatter( y_U_4, x_U_4, c = np.log10(z_U_4), s=1,  )
  plt.colorbar(c)
  ax.set_ylabel(r'Log Temperature $[K]$', fontsize=15 )
  ax.set_xlabel(r'Log Gas Overdensity', fontsize=15 )
  ax.set_title( " CHOLLA Gas GE_TOTAL   ", fontsize=17 )
  ax.set_xlim(x_min, x_max)
  ax.set_ylim(-1, y_max)

  plt.subplot(nrows, ncols, 4*ncols+4)
  ax = plt.gca()
  ax.clear()
  c = ax.scatter( y_GE_4, x_GE_4, c = np.log10(z_GE_4), s=1,  )
  plt.colorbar(c)
  ax.set_ylabel(r'Log Temperature $[K]$', fontsize=15 )
  ax.set_xlabel(r'Log Gas Overdensity', fontsize=15 )
  ax.set_title( " CHOLLA Gas GE_ADVECTED   cell_frac: {0:.3}  mass_frac: {1:.3}".format( cell_frac_4, mass_frac_4), fontsize=17 )
  ax.set_xlim(x_min, x_max)
  ax.set_ylim(-1, y_max)


  plt.subplot(nrows, ncols, 4*ncols+5)
  ax = plt.gca()
  ax.clear()
  c = ax.scatter( y_U_ALL_4, x_U_ALL_4, c = np.log10(z_U_ALL_4), s=1,  )
  plt.colorbar(c)
  ax.set_ylabel(r'Log Temperature $[K]$', fontsize=15 )
  ax.set_xlabel(r'Log Gas Overdensity', fontsize=15 )
  ax.set_title( " CHOLLA Gas GE_TOTAL  ALL ", fontsize=17 )
  ax.set_xlim(x_min, x_max)
  ax.set_ylim(-1, y_max)

  plt.subplot(nrows, ncols, 4*ncols+6)
  ax = plt.gca()
  ax.clear()
  c = ax.scatter( y_GE_ALL_4, x_GE_ALL_4, c = np.log10(z_GE_ALL_4), s=1,  )
  plt.colorbar(c)
  ax.set_ylabel(r'Log Temperature $[K]$', fontsize=15 )
  ax.set_xlabel(r'Log Gas Overdensity', fontsize=15 )
  ax.set_title( " CHOLLA Gas GE_ADVECTED ALL  ", fontsize=17 )
  ax.set_xlim(x_min, x_max)
  ax.set_ylim(-1, y_max)

###########################

# # #
fig.tight_layout()
fig.savefig( outDir + fileName )
print 'Saved image: ', fileName
print ''
  #
