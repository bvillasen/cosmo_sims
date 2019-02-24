import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5

# dataDir = '/raid/bruno/data/cosmo_sims/'
dataDir = '/home/bruno/Desktop/data/'
cosmo_dir = '/home/bruno/Desktop/Dropbox/Developer/cosmo_sims/'
outDir = cosmo_dir + 'figures/phase_diagram/'
toolsDirectory = cosmo_dir + "tools/"
sys.path.extend([toolsDirectory ] )
from load_data_cholla import load_snapshot_data
from load_data_enzo import load_snapshot_enzo
# from cosmo_constants import *
from phase_diagram import get_phase_diagram

dataDir = '/raid/bruno/data/'
outDir = cosmo_dir + 'figures/phase_diagram/'

chollaDir_metals = dataDir + 'cosmo_sims/cholla_pm/256_cool/data_metals/'
chollaDir_uv = dataDir + 'cosmo_sims/cholla_pm/256_cool/data_uv/'

enzoDir_metals = dataDir + 'cosmo_sims/enzo/256_cool_metals/h5_files/'
enzoDir_uv = dataDir + 'cosmo_sims/enzo/256_cool_uv/h5_files/'


gamma = 5./3
nPoints = 256
nx = nPoints
ny = nPoints
nz = nPoints

dv = (115000./nPoints)**3

nSnap = 0
snapshots = range(0,31)
for nSnap in snapshots:
  fileName = 'phase_diagram_{0}.png'.format(nSnap)

  data_cholla = load_snapshot_data( str(nSnap), chollaDir_metals, cool=True)
  current_z_ch = data_cholla['current_z']
  current_a_ch = data_cholla['current_a']
  dens_ch = data_cholla['gas']['density'][...]
  GasEnergy = data_cholla['gas']['GasEnergy'][...] / dens_ch
  temp_ch =  data_cholla['gas']['temperature'][...]
  dens_H_ch = data_cholla['gas']['HI_density'][...]
  dens_mean = dens_ch.mean()
  temp_avrg_ch = temp_ch.mean()
  rho_ch_m = dens_ch.reshape( nx*ny*nz ) / dens_mean
  rho_H_ch_m = dens_H_ch.reshape( nx*ny*nz ) / dens_mean
  temp_ch_m = temp_ch.reshape( nx*ny*nz )
  dens_avrg_ch_m = dens_mean
  dens_H_avrg_ch_m = dens_H_ch.mean()
  temp_avrg_ch_m = temp_avrg_ch


  data_cholla = load_snapshot_data( str(nSnap), chollaDir_uv, cool=True)
  current_z_ch = data_cholla['current_z']
  current_a_ch = data_cholla['current_a']
  dens_ch = data_cholla['gas']['density'][...]
  GasEnergy = data_cholla['gas']['GasEnergy'][...] / dens_ch
  temp_ch =  data_cholla['gas']['temperature'][...]
  dens_H_ch = data_cholla['gas']['HI_density'][...]
  dens_mean = dens_ch.mean()
  temp_avrg_ch = temp_ch.mean()
  rho_ch_uv = dens_ch.reshape( nx*ny*nz ) / dens_mean
  rho_H_ch_uv = dens_H_ch.reshape( nx*ny*nz ) / dens_mean
  temp_ch_uv = temp_ch.reshape( nx*ny*nz )
  dens_avrg_ch_uv = dens_mean
  dens_H_avrg_ch_uv = dens_H_ch.mean()
  temp_avrg_ch_uv = temp_avrg_ch

  data_enzo = load_snapshot_enzo( nSnap, enzoDir_metals, dm=False, cool=True)
  current_z_en = data_enzo['current_z']
  dens_en = data_enzo['gas']['density'][...]
  GasEnergy = data_enzo['gas']['GasEnergy'][...] / dens_en
  temp_en =  data_enzo['gas']['temperature'][...]
  dens_H_en = data_enzo['gas']['HI_density'][...]
  dens_mean = dens_en.mean()
  temp_avrg_en = temp_en.mean()
  rho_en_m = dens_en.reshape( nx*ny*nz ) / dens_mean
  rho_H_en_m = dens_H_en.reshape( nx*ny*nz ) / dens_mean
  temp_en_m = temp_en.reshape( nx*ny*nz )
  dens_avrg_en_m = dens_mean
  dens_H_avrg_en_m = dens_H_en.mean()
  temp_avrg_en_m = temp_avrg_en

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


  nbins = 2000
  x_ch_m, y_ch_m, z_ch_m = get_phase_diagram( rho_ch_m, temp_ch_m , nbins )
  x_H_ch_m, y_H_ch_m, z_H_ch_m = get_phase_diagram( rho_H_ch_m, temp_ch_m , nbins )
  x_ch_uv, y_ch_uv, z_ch_uv = get_phase_diagram( rho_ch_uv, temp_ch_uv , nbins )
  x_H_ch_uv, y_H_ch_uv, z_H_ch_uv = get_phase_diagram( rho_H_ch_uv, temp_ch_uv , nbins )
  x_en_m, y_en_m, z_en_m = get_phase_diagram( rho_en_m, temp_en_m , nbins )
  x_H_en_m, y_H_en_m, z_H_en_m = get_phase_diagram( rho_H_en_m, temp_en_m , nbins )
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
  c = ax.scatter( y_en_m, x_en_m, c = np.log10(z_en_m), s=1,  )
  plt.colorbar(c)
  ax.set_ylabel(r'Log Temperature $[K]$', fontsize=15 )
  ax.set_xlabel(r'Log Gas Overdensity', fontsize=15 )
  ax.set_title( "Z={0:.2f}        ENZO Gas NO UVB  T={1:.2e}   d={2:.2e}".format(current_z_ch, temp_avrg_en_m, dens_avrg_en_m))
  ax.set_xlim(x_min, x_max)
  ax.set_ylim(-1, y_max)

  plt.subplot(2, 4, 2)
  ax = plt.gca()
  ax.clear()
  c = ax.scatter( y_H_en_m, x_H_en_m, c = np.log10(z_H_en_m), s=1,  )
  plt.colorbar(c)
  ax.set_ylabel(r'Log Temperature $[K]$', fontsize=15 )
  ax.set_xlabel(r'Log Neutral Hydrogen Overdensity', fontsize=15 )
  ax.set_title( " ENZO Neutral Hydrogen NO UVB  T={0:.2e}   d={1:.2e}".format( temp_avrg_en_m, dens_H_avrg_en_m) )
  ax.set_xlim(x_min_h, x_max_h)
  ax.set_ylim(-1, y_max)

  plt.subplot(2, 4, 3)
  ax = plt.gca()
  ax.clear()
  c = ax.scatter( y_en_uv, x_en_uv, c = np.log10(z_en_uv), s=1,  )
  plt.colorbar(c)
  ax.set_ylabel(r'Log Temperature $[K]$', fontsize=15 )
  ax.set_xlabel(r'Log Gas Overdensity', fontsize=15 )
  ax.set_title( " ENZO Gas UVB HM2012  T={0:.2e}   d={1:.2e}".format( temp_avrg_en_uv, dens_avrg_en_uv))
  ax.set_xlim(x_min, x_max)
  ax.set_ylim(-1, y_max)

  plt.subplot(2, 4, 4)
  ax = plt.gca()
  ax.clear()
  c = ax.scatter( y_H_en_uv, x_H_en_uv, c = np.log10(z_H_en_uv), s=1,  )
  plt.colorbar(c)
  ax.set_ylabel(r'Log Temperature $[K]$', fontsize=15 )
  ax.set_xlabel(r'Log Neutral Hydrogen Overdensity', fontsize=15 )
  ax.set_title( " ENZO Neutral Hydrogen UVB HM2012  T={0:.2e}   d={1:.2e}".format( temp_avrg_en_uv, dens_H_avrg_en_uv))
  ax.set_xlim(x_min_h, x_max_h)
  ax.set_ylim(-1, y_max)


  plt.subplot(2, 4, 5)
  ax = plt.gca()
  ax.clear()
  c = ax.scatter( y_ch_m, x_ch_m, c = np.log10(z_ch_m), s=1,  )
  plt.colorbar(c)
  ax.set_ylabel(r'Log Temperature $[K]$', fontsize=15 )
  ax.set_xlabel(r'Log Gas Overdensity', fontsize=15 )
  ax.set_title( " CHOLLA Gas NO UVB  T={0:.2e}   d={1:.2e}".format( temp_avrg_ch_m, dens_avrg_ch_m))
  ax.set_xlim(x_min, x_max)
  ax.set_ylim(-1, y_max)

  plt.subplot(2, 4, 6)
  ax = plt.gca()
  ax.clear()
  c = ax.scatter( y_H_ch_m, x_H_ch_m, c = np.log10(z_H_ch_m), s=1,  )
  plt.colorbar(c)
  ax.set_ylabel(r'Log Temperature $[K]$', fontsize=15 )
  ax.set_xlabel(r'Log Neutral Hydrogen Overdensity', fontsize=15 )
  ax.set_title( " CHOLLA Neutral Hydrogen NO UVB  T={0:.2e}   d={1:.2e}".format( temp_avrg_ch_m, dens_H_avrg_ch_m))
  ax.set_xlim(x_min_h, x_max_h)
  ax.set_ylim(-1, y_max)


  plt.subplot(2, 4, 7)
  ax = plt.gca()
  ax.clear()
  c = ax.scatter( y_ch_uv, x_ch_uv, c = np.log10(z_ch_uv), s=1,  )
  plt.colorbar(c)
  ax.set_ylabel(r'Log Temperature $[K]$', fontsize=15 )
  ax.set_xlabel(r'Log Gas Overdensity', fontsize=15 )
  ax.set_title( " CHOLLA Gas UVB HM2012  T={0:.2e}   d={1:.2e}".format( temp_avrg_ch_uv, dens_avrg_ch_uv))
  ax.set_xlim(x_min, x_max)
  ax.set_ylim(-1, y_max)

  plt.subplot(2, 4, 8)
  ax = plt.gca()
  ax.clear()
  c = ax.scatter( y_H_ch_uv, x_H_ch_uv, c = np.log10(z_H_ch_uv), s=1,  )
  plt.colorbar(c)
  ax.set_ylabel(r'Log Temperature $[K]$', fontsize=15 )
  ax.set_xlabel(r'Log Neutral Hydrogen Overdensity', fontsize=15 )
  ax.set_title( " CHOLLA Neutral Hydrogen UVB HM2012  T={0:.2e}   d={1:.2e}".format( temp_avrg_ch_uv, dens_H_avrg_ch_uv))
  ax.set_xlim(x_min_h, x_max_h)
  ax.set_ylim(-1, y_max)

  fig.tight_layout()
  fig.savefig( outDir + fileName )
  print 'Saved image: ', fileName
  print ''
#
#
#   # plt.figure(0)
#   # plt.clf()
#   #
#   #
#   # t_cosmo = t_start * ( 1 + current_z_ch_1 )**2
#   # plt.plot( [ n_list[0], n_list[-1] ] , [t_cosmo, t_cosmo],'--', c='r'  )
#   # plt.plot( n_list, temps_average_v,  label='mean_volume' )
#   # plt.plot( n_list, temps_average_m,  label='mean_mass' )
#   # plt.plot( n_list, temps_median_v,  label='median_volume' )
#   # plt.plot( n_list, temps_median_m,  label='median_mass' )
#   # plt.xlabel( 'N steps' )
#   # plt.ylabel( r'Temperature  [K]')
#   # plt.legend(loc=1)
#   # plt.xscale('log')
#   # # plt.xscale('log')
#   # plt.ylim( 100, 180)
#   # plt.legend(loc=4)
#   # plt.title( 'Gravity OFF')
#   # plt.savefig(outDir + 'temperature_statistics_{0}_nograv.png'.format(nSnap))
