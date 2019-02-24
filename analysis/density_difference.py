import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
from power_spectrum import get_power_spectrum

cosmo_dir = '/home/bruno/Desktop/Dropbox/Developer/cosmo_sims/'
toolsDirectory = cosmo_dir + "tools/"
sys.path.extend([toolsDirectory ] )
from load_data_cholla import load_snapshot_data
from load_data_enzo import load_snapshot_enzo


dataDir = '/raid/bruno/data/'
chollaDir = dataDir + 'cosmo_sims/cholla_pm/256_cool/data_metals/'
enzoDir = dataDir + 'cosmo_sims/enzo/256_cool_metals/h5_files/'

outDir = cosmo_dir + 'figures/density_difference/'



def get_density_differesnce_distribution( dens, diff, nbins, y_min, y_max ):
  nz, ny, nx = dens.shape
  diff = diff.reshape( nz * ny * nx )
  dens = dens.reshape( nz * ny * nx ) / dens.mean()
  # indxs = np.where( (diff < y_max) and (diff > y_min) )
  indxs = ( diff > y_min) * ( diff < y_max )
  diff = diff[indxs]
  dens = dens[indxs]
  n_total = len(dens)

  phase, yedges, xedges  = np.histogram2d( np.log10(dens), diff, bins=nbins, density=False )
  xcenters = (xedges[:-1] + xedges[1:])/2
  ycenters = (yedges[:-1] + yedges[1:])/2
  X, Y = np.meshgrid( xcenters, ycenters )
  x = X.flatten()
  y = Y.flatten()
  z = phase.flatten()
  indxs = np.where(z>0)
  x = x[indxs]
  y = y[indxs]
  z = z[indxs]
  return x, y, z, n_total

nbins = 1000


x_min, x_max = -5, 3
y_min, y_max = -1.1, 1.5

nSnap = 0
snapshots = range(0, 31)
for i,nSnap in enumerate(snapshots):
  fileName = 'density_difference_{0}.png'.format(nSnap)

  data_cholla = load_snapshot_data( nSnap, chollaDir, cool=True )
  current_z_ch = data_cholla['current_z']
  dens_dm_ch = data_cholla['dm']['density'][...]
  dens_ch = data_cholla['gas']['density'][...]
  HI_ch = data_cholla['gas']['HI_density'][...]
  HII_ch = data_cholla['gas']['HII_density'][...]
  delta_ch = ( HI_ch - HI_ch.mean() ) / HI_ch

  data_enzo = load_snapshot_enzo( nSnap, enzoDir, dm=True, cool=True)
  current_a_enzo = data_enzo['current_a']
  current_z_enzo = data_enzo['current_z']
  dens_dm_en = data_enzo['dm']['density'][...]
  dens_en = data_enzo['gas']['density'][...]
  HI_en = data_enzo['gas']['HI_density'][...]
  HII_en = data_enzo['gas']['HII_density'][...]

  diff_0 = ( dens_ch - dens_en ) / dens_en
  diff_1 = ( HI_ch - HI_en ) / HI_en
  diff_2 = ( HII_ch - HII_en ) / HII_en

  x_0, y_0, z_0, n_0 = get_density_differesnce_distribution( dens_en, diff_0, nbins, y_min, y_max )
  x_1, y_1, z_1, n_1 = get_density_differesnce_distribution( HI_en, diff_1, nbins, y_min, y_max )
  x_2, y_2, z_2, n_2 = get_density_differesnce_distribution( HII_en, diff_2, nbins, y_min, y_max )

  fig, ax_l = plt.subplots(nrows=1, ncols=3, figsize=(30,10))

  plt.subplot(1, 3, 1)
  ax = plt.gca()
  ax.clear()
  c = ax.scatter( y_0, x_0, c = np.log10(z_0/n_0), s=1,  )
  plt.colorbar(c)
  ax.axhline( y=0., color='r', linestyle='--',  )
  ax.set_xlim( x_min, x_max )
  ax.set_ylim( y_min, y_max )
  ax.set_ylabel(r'Cell Density Difference', fontsize=15 )
  ax.set_xlabel(r'Log Gas Overdensity', fontsize=15 )
  ax.set_title( "Z={0:.2f}   Gas Density Difference".format(current_z_ch ) )

  plt.subplot(1, 3, 2)
  ax = plt.gca()
  ax.clear()
  c = ax.scatter( y_1, x_1, c = np.log10(z_1/n_1), s=1,  )
  plt.colorbar(c)
  ax.axhline( y=0., color='r', linestyle='--',  )
  ax.set_xlim( x_min, x_max )
  ax.set_ylim( y_min, y_max )
  ax.set_xlabel(r'Log Neutral Hydrogen Overdensity', fontsize=15 )
  ax.set_title( "Neutral Hydrogen Density Difference".format(current_z_ch ) )

  plt.subplot(1, 3, 3)
  ax = plt.gca()
  ax.clear()
  c = ax.scatter( y_2, x_2, c = np.log10(z_2/n_2), s=1,  )
  plt.colorbar(c)
  ax.axhline( y=0., color='r', linestyle='--',  )
  ax.set_xlim( x_min, x_max )
  ax.set_ylim( y_min, y_max )
  ax.set_xlabel(r'Log Ionized Hydrogen Overdensity', fontsize=15 )
  ax.set_title( "Ionized Hydrogen Density Difference".format(current_z_ch ) )

  fig.tight_layout()
  fig.savefig( outDir + fileName )
  print 'Saved image: ', fileName
  print ''
