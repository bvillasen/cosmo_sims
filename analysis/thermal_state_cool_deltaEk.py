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
# from cosmo_constants import *
from phase_diagram import get_phase_diagram

dataDir = '/raid/bruno/data/'
outDir = cosmo_dir + 'figures/phase_diagram/deltaEk/'

chollaDir_metals = dataDir + 'cosmo_sims/cholla_pm/256_cool/data_metals/'
chollaDir_0 = dataDir + 'cosmo_sims/cholla_pm/256_cool/data_de02_limitEkin/'
chollaDir_1 = dataDir + 'cosmo_sims/cholla_pm/256_cool/data_de02_gravWork/'
chollaDir_2 = dataDir + 'cosmo_sims/cholla_pm/256_cool/data_de02_deltaEk/'



enzoDir_metals = dataDir + 'cosmo_sims/enzo/256_cool_metals/h5_files/'
enzoDir_uv = dataDir + 'cosmo_sims/enzo/256_cool_uv/h5_files/'

plot_metals = False

gamma = 5./3
nPoints = 256
nx = nPoints
ny = nPoints
nz = nPoints

dv = (115000./nPoints)**3
nbins = 1000

nSnap = 0
snapshots = range(16,31)
# snapshots = [0]
for nSnap in snapshots:
  fileName = 'phase_diagram_{0}.png'.format(nSnap)

  data_cholla = load_snapshot_data( str(nSnap), chollaDir_0, cool=True)
  current_z_ch = data_cholla['current_z']
  current_a_ch = data_cholla['current_a']
  dens_ch = data_cholla['gas']['density'][...]
  # GasEnergy = data_cholla['gas']['GasEnergy'][...] / dens_ch
  temp_ch =  data_cholla['gas']['temperature'][...]
  dens_H_ch = data_cholla['gas']['HI_density'][...]
  dens_mean = dens_ch.mean()
  temp_avrg_ch = temp_ch.mean()
  rho_ch_uv = dens_ch.reshape( nx*ny*nz ) / dens_mean
  rho_H_ch_uv = dens_H_ch.reshape( nx*ny*nz ) / dens_mean
  temp_ch_uv = temp_ch.reshape( nx*ny*nz )
  dens_avrg_ch_uv_0 = dens_mean
  dens_H_avrg_ch_uv_0 = dens_H_ch.mean()
  temp_avrg_ch_uv_0 = temp_avrg_ch

  x_ch_uv_0, y_ch_uv_0, z_ch_uv_0 = get_phase_diagram( rho_ch_uv, temp_ch_uv , nbins )
  x_H_ch_uv_0, y_H_ch_uv_0, z_H_ch_uv_0 = get_phase_diagram( rho_H_ch_uv, temp_ch_uv , nbins )


  data_cholla = load_snapshot_data( str(nSnap), chollaDir_1, cool=True)
  current_z_ch = data_cholla['current_z']
  current_a_ch = data_cholla['current_a']
  dens_ch = data_cholla['gas']['density'][...]
  # GasEnergy = data_cholla['gas']['GasEnergy'][...] / dens_ch
  temp_ch =  data_cholla['gas']['temperature'][...]
  dens_H_ch = data_cholla['gas']['HI_density'][...]
  dens_mean = dens_ch.mean()
  temp_avrg_ch = temp_ch.mean()
  rho_ch_uv = dens_ch.reshape( nx*ny*nz ) / dens_mean
  rho_H_ch_uv = dens_H_ch.reshape( nx*ny*nz ) / dens_mean
  temp_ch_uv = temp_ch.reshape( nx*ny*nz )
  dens_avrg_ch_uv_1 = dens_mean
  dens_H_avrg_ch_uv_1 = dens_H_ch.mean()
  temp_avrg_ch_uv_1 = temp_avrg_ch

  x_ch_uv_1, y_ch_uv_1, z_ch_uv_1 = get_phase_diagram( rho_ch_uv, temp_ch_uv , nbins )
  x_H_ch_uv_1, y_H_ch_uv_1, z_H_ch_uv_1 = get_phase_diagram( rho_H_ch_uv, temp_ch_uv , nbins )


  data_cholla = load_snapshot_data( str(nSnap), chollaDir_2, cool=True)
  current_z_ch = data_cholla['current_z']
  current_a_ch = data_cholla['current_a']
  dens_ch = data_cholla['gas']['density'][...]
  # GasEnergy = data_cholla['gas']['GasEnergy'][...] / dens_ch
  temp_ch =  data_cholla['gas']['temperature'][...]
  dens_H_ch = data_cholla['gas']['HI_density'][...]
  dens_mean = dens_ch.mean()
  temp_avrg_ch = temp_ch.mean()
  rho_ch_uv = dens_ch.reshape( nx*ny*nz ) / dens_mean
  rho_H_ch_uv = dens_H_ch.reshape( nx*ny*nz ) / dens_mean
  temp_ch_uv = temp_ch.reshape( nx*ny*nz )
  dens_avrg_ch_uv_2 = dens_mean
  dens_H_avrg_ch_uv_2 = dens_H_ch.mean()
  temp_avrg_ch_uv_2 = temp_avrg_ch

  x_ch_uv_2, y_ch_uv_2, z_ch_uv_2 = get_phase_diagram( rho_ch_uv, temp_ch_uv , nbins )
  x_H_ch_uv_2, y_H_ch_uv_2, z_H_ch_uv_2 = get_phase_diagram( rho_H_ch_uv, temp_ch_uv , nbins )



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
  x_en_uv, y_en_uv, z_en_uv = get_phase_diagram( rho_en_uv, temp_en_uv , nbins )
  x_H_en_uv, y_H_en_uv, z_H_en_uv = get_phase_diagram( rho_H_en_uv, temp_en_uv , nbins )

  fig, ax_l = plt.subplots(nrows=2, ncols=4, figsize=(40,16))
  x_min = -2
  x_max = 3
  x_min_h = -11
  x_max_h = 3

  y_max = 8


  plt.subplot(2, 4, 1)
  ax = plt.gca()
  ax.clear()
  c = ax.scatter( y_en_uv, x_en_uv, c = np.log10(z_en_uv), s=1,  )
  plt.colorbar(c)
  ax.set_ylabel(r'Log Temperature $[K]$', fontsize=15 )
  ax.set_xlabel(r'Log Gas Overdensity', fontsize=15 )
  ax.set_title( " ENZO Gas UVB HM2012  T={0:.2e}   d={1:.2e}".format( temp_avrg_en_uv, dens_avrg_en_uv))
  ax.set_xlim(x_min, x_max)
  ax.set_ylim(-1, y_max)

  plt.subplot(2, 4, 5)
  ax = plt.gca()
  ax.clear()
  c = ax.scatter( y_H_en_uv, x_H_en_uv, c = np.log10(z_H_en_uv), s=1,  )
  plt.colorbar(c)
  ax.set_ylabel(r'Log Temperature $[K]$', fontsize=15 )
  ax.set_xlabel(r'Log Neutral Hydrogen Overdensity', fontsize=15 )
  ax.set_title( " ENZO Neutral Hydrogen UVB HM2012  T={0:.2e}   d={1:.2e}".format( temp_avrg_en_uv, dens_H_avrg_en_uv))
  ax.set_xlim(x_min_h, x_max_h)
  ax.set_ylim(-1, y_max)



  plt.subplot(2, 4, 2)
  ax = plt.gca()
  ax.clear()
  c = ax.scatter( y_ch_uv_0, x_ch_uv_0, c = np.log10(z_ch_uv_0), s=1,  )
  plt.colorbar(c)
  ax.set_ylabel(r'Log Temperature $[K]$', fontsize=15 )
  ax.set_xlabel(r'Log Gas Overdensity', fontsize=15 )
  ax.set_title( " CHOLLA Gas LIMIT EKIN  T={0:.2e}   d={1:.2e}".format( temp_avrg_ch_uv_0, dens_avrg_ch_uv_0))
  ax.set_xlim(x_min, x_max)
  ax.set_ylim(-1, y_max)

  plt.subplot(2, 4, 6)
  ax = plt.gca()
  ax.clear()
  c = ax.scatter( y_H_ch_uv_0, x_H_ch_uv_0, c = np.log10(z_H_ch_uv_0), s=1,  )
  plt.colorbar(c)
  ax.set_ylabel(r'Log Temperature $[K]$', fontsize=15 )
  ax.set_xlabel(r'Log Neutral Hydrogen Overdensity', fontsize=15 )
  ax.set_title( " CHOLLA Neutral Hydrogen LIMIT EKIN  T={0:.2e}   d={1:.2e}".format( temp_avrg_ch_uv_0, dens_H_avrg_ch_uv_0))
  ax.set_xlim(x_min_h, x_max_h)
  ax.set_ylim(-1, y_max)

  plt.subplot(2, 4, 3)
  ax = plt.gca()
  ax.clear()
  c = ax.scatter( y_ch_uv_1, x_ch_uv_1, c = np.log10(z_ch_uv_1), s=1,  )
  plt.colorbar(c)
  ax.set_ylabel(r'Log Temperature $[K]$', fontsize=15 )
  ax.set_xlabel(r'Log Gas Overdensity', fontsize=15 )
  ax.set_title( " CHOLLA Gas GRAVITATIONAL WORK  T={0:.2e}   d={1:.2e}".format( temp_avrg_ch_uv_1, dens_avrg_ch_uv_1))
  ax.set_xlim(x_min, x_max)
  ax.set_ylim(-1, y_max)

  plt.subplot(2, 4, 7)
  ax = plt.gca()
  ax.clear()
  c = ax.scatter( y_H_ch_uv_1, x_H_ch_uv_1, c = np.log10(z_H_ch_uv_1), s=1,  )
  plt.colorbar(c)
  ax.set_ylabel(r'Log Temperature $[K]$', fontsize=15 )
  ax.set_xlabel(r'Log Neutral Hydrogen Overdensity', fontsize=15 )
  ax.set_title( " CHOLLA Neutral Hydrogen GRAVITATIONAL WORK  T={0:.2e}   d={1:.2e}".format( temp_avrg_ch_uv_1, dens_H_avrg_ch_uv_1))
  ax.set_xlim(x_min_h, x_max_h)
  ax.set_ylim(-1, y_max)

  plt.subplot(2, 4, 4)
  ax = plt.gca()
  ax.clear()
  c = ax.scatter( y_ch_uv_2, x_ch_uv_2, c = np.log10(z_ch_uv_2), s=1,  )
  plt.colorbar(c)
  ax.set_ylabel(r'Log Temperature $[K]$', fontsize=15 )
  ax.set_xlabel(r'Log Gas Overdensity', fontsize=15 )
  ax.set_title( " CHOLLA Gas GRAVITY DELTA EKIN  T={0:.2e}   d={1:.2e}".format( temp_avrg_ch_uv_2, dens_avrg_ch_uv_2))
  ax.set_xlim(x_min, x_max)
  ax.set_ylim(-1, y_max)

  plt.subplot(2, 4, 8)
  ax = plt.gca()
  ax.clear()
  c = ax.scatter( y_H_ch_uv_2, x_H_ch_uv_2, c = np.log10(z_H_ch_uv_2), s=1,  )
  plt.colorbar(c)
  ax.set_ylabel(r'Log Temperature $[K]$', fontsize=15 )
  ax.set_xlabel(r'Log Neutral Hydrogen Overdensity', fontsize=15 )
  ax.set_title( " CHOLLA Neutral Hydrogen GRAVITY  DELTA EKIN  T={0:.2e}   d={1:.2e}".format( temp_avrg_ch_uv_2, dens_H_avrg_ch_uv_2))
  ax.set_xlim(x_min_h, x_max_h)
  ax.set_ylim(-1, y_max)



  fig.tight_layout()
  fig.savefig( outDir + fileName )
  print 'Saved image: ', fileName
  print ''
#
