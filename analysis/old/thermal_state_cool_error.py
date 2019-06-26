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

dataDir = '/raid/bruno/data/'
outDir = cosmo_dir + 'figures/phase_diagram/uvb_error_cells_1/'

chollaDir_0 = dataDir + 'cosmo_sims/cholla_pm/256_cool/data_de02_error_cells/'
# chollaDir_1 = dataDir + 'cosmo_sims/cholla_pm/256_cool/data_de02_limitChangeDE/'
# chollaDir_2 = dataDir + 'cosmo_sims/cholla_pm/256_cool/data_de02_grav2/'


enzoDir_uv = dataDir + 'cosmo_sims/enzo/256_cool_uv/h5_files/'


gamma = 5./3
nPoints = 256
nx = nPoints
ny = nPoints
nz = nPoints
ncells = nx * ny * nz

dv = (115000./nPoints)**3
nbins = 1000


# err_indxs_tr = (np.array([ 67,  92, 150, 160, 161, 171]),
#  np.array([242,  28, 252,  66,  10, 192]),
#  np.array([190, 190,   7, 100, 240, 176]))

err_indxs_tr = (np.array([ 67,  92, 160, 161, 171]),
 np.array([242,  28,  66,  10, 192]),
 np.array([190, 190, 100, 240, 176]))

 
# n_err = len(err_indxs_tr[0])
# err_indxs = []
# for i in range(n_err):
#   k = err_indxs_tr[0][i]
#   j = err_indxs_tr[1][i]
#   i = err_indxs_tr[2][i]
#   err_indx = ( np.array([k]), np.array([j]), np.array([i]) )
#   err_indxs.append( err_indx )



nSnap = 27
snapshots = range(10,28)
for nSnap in snapshots:
  fileName = 'phase_diagram_{0}.png'.format(nSnap)

  data_cholla = load_snapshot_data( str(nSnap), chollaDir_0, cool=True)
  current_z_ch = data_cholla['current_z']
  current_a_ch = data_cholla['current_a']
  dens = data_cholla['gas']['density'][...]
  dens_H = data_cholla['gas']['HI_density'][...]
  dens_mean = dens.mean()
  temp_GK, temp_DE, temp_U, temp_GE, dens_U, dens_GE, HI_dens_U, HI_dens_GE, temp_U_ALL, temp_GE_ALL =  get_Temperaure_From_Flags_DE( data_cholla, gamma=5./3, normalize_dens=True, )
  x_GK_0, y_GK_0, z_GK_0 = get_phase_diagram( dens.flatten()/dens_mean, temp_GK , nbins, ncells )
  x_U_0, y_U_0, z_U_0 = get_phase_diagram( dens_U, temp_U , nbins, ncells )
  x_GE_0, y_GE_0, z_GE_0 = get_phase_diagram( dens_GE, temp_GE , nbins, ncells )
  cell_frac_0 = float(z_GE_0.sum()) 
  mass_frac_0 = dens_GE.sum()*dens_mean/dens.sum()

  err_dens = np.log10( dens[err_indxs_tr] / dens_mean )
  err_temp = np.log10( temp_DE[err_indxs_tr] )
  
  
  
  ncols = 3
  nrows = 1
  fig, ax_l = plt.subplots(nrows=2, ncols=ncols, figsize=(10*ncols,8*nrows))
  x_min = -2
  x_max = 5
  x_min_h = -11
  x_max_h = 3
  
  y_max = 8
  
  
  plt.subplot(nrows, ncols, 1)
  ax = plt.gca()
  ax.clear()
  c = ax.scatter( y_GK_0, x_GK_0, c = np.log10(z_GK_0), s=1,  )
  plt.colorbar(c)
  ax.set_ylabel(r'Log Temperature $[K]$', fontsize=15 )
  ax.set_xlabel(r'Log Gas Overdensity', fontsize=15 )
  ax.set_title( " CHOLLA Gas GE_GRACKLE   ",fontsize=17 )
  ax.set_xlim(x_min, x_max)
  ax.set_ylim(-1, y_max)
  # 
  # 
  plt.subplot(nrows, ncols, 2)
  ax = plt.gca()
  ax.clear()
  c = ax.scatter( y_U_0, x_U_0, c = np.log10(z_U_0), s=1,  )
  plt.colorbar(c)
  ax.set_ylabel(r'Log Temperature $[K]$', fontsize=15 )
  ax.set_xlabel(r'Log Gas Overdensity', fontsize=15 )
  ax.set_title( " CHOLLA Gas GE_TOTAL   ", fontsize=17 )
  ax.set_xlim(x_min, x_max)
  ax.set_ylim(-1, y_max)
  # 
  plt.subplot(nrows, ncols, 3)
  ax = plt.gca()
  ax.clear()
  c = ax.scatter( y_GE_0, x_GE_0, c = np.log10(z_GE_0), s=1,  )
  plt.colorbar(c)
  # c = ax.scatter( err_dens, err_temp, c = 'r', s=5,  )
  ax.set_ylabel(r'Log Temperature $[K]$', fontsize=15 )
  ax.set_xlabel(r'Log Gas Overdensity', fontsize=15 )
  ax.set_title( " CHOLLA Gas GE_ADVECTED   cell_frac: {0:.3}  mass_frac: {1:.3}".format( cell_frac_0, mass_frac_0), fontsize=17 )
  ax.set_xlim(x_min, x_max)
  ax.set_ylim(-1, y_max)
  
  # 
  # 
  
  # # # 
  fig.tight_layout()
  fig.savefig( outDir + fileName )
  print 'Saved image: ', fileName
  print ''
    #
