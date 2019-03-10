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

dataSet = 'de1'
chollaDir = dataDir + 'cosmo_sims/cholla_pm/256_cool/data_{0}/'.format(dataSet)
enzoDir = dataDir + 'cosmo_sims/enzo/256_cool_uv/h5_files/'
outDir = cosmo_dir + 'figures/power_hydro/'

fileName = outDir + 'ps_256_cooling_uv_{0}.png'.format( dataSet )

# set simulation volume dimentions
nPoints = 256
nz, ny, nx = nPoints, nPoints, nPoints
nCells  = nx*ny*nz
Lbox = 115.0   #Mpc/h
h = 0.6774
Lx = Lbox
Ly = Lbox
Lz = Lbox
dx, dy, dz = Lx/(nx), Ly/(ny), Lz/(nz )

n_kSamples = 20



snapshots = [ 0, 2, 4, 7, 10, 13, 16, 20, 25, 30]
# snapshots = [ 0, ]
fig = plt.figure(0)
fig.set_size_inches(40,10)
fig.clf()


gs = plt.GridSpec(5, 4)
gs.update(hspace=0.05, wspace=0.08, )
ax1 = plt.subplot(gs[0:4, 0])
ax2 = plt.subplot(gs[4:5, 0])

ax3 = plt.subplot(gs[0:4, 1])
ax4 = plt.subplot(gs[4:5, 1])

ax5 = plt.subplot(gs[0:4, 2])
ax6 = plt.subplot(gs[4:5, 2])

ax7 = plt.subplot(gs[0:4, 3])
ax8 = plt.subplot(gs[4:5, 3])


colors = ['b', 'y', 'g', 'c', 'm', 'b', 'y', 'g', 'c', 'm', ]

for i,nSnap in enumerate(snapshots):
  print " Cholla: ", nSnap
  snapKey = str( nSnap )
  # if i not in [9]: continue
  data_cholla = load_snapshot_data( snapKey, chollaDir, cool=True )
  current_z_ch = data_cholla['current_z']
  dens_dm_cholla = data_cholla['dm']['density'][...]
  dens_gas_cholla = data_cholla['gas']['density'][...]
  dens_gas_H_cholla = data_cholla['gas']['HI_density'][...]
  dens_gas_HII_cholla = data_cholla['gas']['HII_density'][...]

  ps_dm_cholla, k_vals, count_dm_cholla = get_power_spectrum( dens_dm_cholla, Lbox, nx, ny, nz, dx, dy, dz,  n_kSamples=n_kSamples)
  ps_gas_cholla, k_vals, count_gas_cholla = get_power_spectrum( dens_gas_cholla, Lbox, nx, ny, nz, dx, dy, dz,  n_kSamples=n_kSamples)
  ps_gas_H_cholla, k_vals, count_gas_cholla = get_power_spectrum( dens_gas_H_cholla, Lbox, nx, ny, nz, dx, dy, dz,  n_kSamples=n_kSamples)
  ps_gas_HII_cholla, k_vals, count_gas_cholla = get_power_spectrum( dens_gas_HII_cholla, Lbox, nx, ny, nz, dx, dy, dz,  n_kSamples=n_kSamples)

  c= colors[i]
  label = 'z = {0:.1f}'.format(current_z_ch)
  ax1.plot( k_vals, ps_dm_cholla,  c=c, linewidth=2, label=label )
  ax3.plot( k_vals, ps_gas_cholla, c=c,  linewidth=2, label=label )
  ax5.plot( k_vals, ps_gas_H_cholla, c=c,  linewidth=2, label=label )
  ax7.plot( k_vals, ps_gas_HII_cholla, c=c,  linewidth=2, label=label )

  print ' Enzo: ', nSnap
  data_enzo = load_snapshot_enzo( nSnap, enzoDir, dm=True, cool=True)
  current_a_enzo = data_enzo['current_a']
  current_z_enzo = data_enzo['current_z']
  dens_dm_enzo = data_enzo['dm']['density'][...]
  dens_gas_enzo = data_enzo['gas']['density'][...]
  dens_gas_H_enzo = data_enzo['gas']['HI_density'][...]
  dens_gas_HII_enzo = data_enzo['gas']['HII_density'][...]

  ps_dm_enzo, k_vals, count_dm_enzo = get_power_spectrum( dens_dm_enzo, Lbox, nx, ny, nz, dx, dy, dz,  n_kSamples=n_kSamples)
  ps_gas_enzo, k_vals, count_gas_enzo = get_power_spectrum( dens_gas_enzo, Lbox, nx, ny, nz, dx, dy, dz,  n_kSamples=n_kSamples)
  ps_gas_H_enzo, k_vals, count_gas_enzo = get_power_spectrum( dens_gas_H_enzo, Lbox, nx, ny, nz, dx, dy, dz,  n_kSamples=n_kSamples)
  ps_gas_HII_enzo, k_vals, count_gas_enzo = get_power_spectrum( dens_gas_HII_enzo, Lbox, nx, ny, nz, dx, dy, dz,  n_kSamples=n_kSamples)

  # label = 'z = {0:.1f}'.format(current_z_enzo)
  if i == 9:
    ax1.plot( k_vals, ps_dm_enzo, '--', c=c, linewidth=1, label='Enzo' )
    ax3.plot( k_vals, ps_gas_enzo, '--', c=c, linewidth=1, label='Enzo' )
    ax5.plot( k_vals, ps_gas_H_enzo, '--', c=c, linewidth=1, label='Enzo' )
    ax7.plot( k_vals, ps_gas_HII_enzo, '--', c=c, linewidth=1, label='Enzo' )
  else:
    ax1.plot( k_vals, ps_dm_enzo, '--', c=c, linewidth=1 )
    ax3.plot( k_vals, ps_gas_enzo, '--', c=c, linewidth=1 )
    ax5.plot( k_vals, ps_gas_H_enzo, '--', c=c, linewidth=1  )
    ax7.plot( k_vals, ps_gas_HII_enzo, '--', c=c, linewidth=1 )
  #

  error_dm = (ps_dm_cholla - ps_dm_enzo) / ps_dm_cholla
  error_gas = (ps_gas_cholla - ps_gas_enzo) / ps_gas_cholla
  error_gas_H = (ps_gas_H_cholla - ps_gas_H_enzo) / ps_gas_H_cholla
  error_gas_HII = (ps_gas_HII_cholla - ps_gas_HII_enzo) / ps_gas_HII_cholla

  ax2.plot( k_vals, error_dm , c=c, alpha=0.9)
  ax4.plot( k_vals, error_gas , c=c, alpha=0.9)
  ax6.plot( k_vals, error_gas_H , c=c, alpha=0.9)
  ax8.plot( k_vals, error_gas_HII , c=c, alpha=0.9)

ax2.axhline( y=0., color='r', linestyle='--',  )
ax2.set_ylim( -0.4, 0.4)
ax4.axhline( y=0., color='r', linestyle='--',  )
ax4.set_ylim( -0.4, 0.4)
ax6.axhline( y=0., color='r', linestyle='--',  )
ax6.set_ylim( -0.4, 0.4)
ax8.axhline( y=0., color='r', linestyle='--',  )
ax8.set_ylim( -0.4, 0.4)

ax1.set_ylabel( r'$P(k) $', fontsize=17)
ax2.set_ylabel( 'Difference', fontsize=15)
ax1.legend( loc=3)
ax2.set_xlabel( r'$k \, \, [h Mpc^{-1}]$', fontsize=17)
ax3.legend( loc=3)
ax4.set_xlabel( r'$k \, \, [h Mpc^{-1}]$', fontsize=17)
ax5.legend( loc=3)
ax6.set_xlabel( r'$k \, \, [h Mpc^{-1}]$', fontsize=17)
ax7.legend( loc=3)
ax8.set_xlabel( r'$k \, \, [h Mpc^{-1}]$', fontsize=17)

ax1.set_xscale('log')
ax1.set_yscale('log')
ax3.set_xscale('log')
ax3.set_yscale('log')
ax5.set_xscale('log')
ax5.set_yscale('log')
ax7.set_xscale('log')
ax7.set_yscale('log')
ax2.set_xscale('log')
ax4.set_xscale('log')
ax6.set_xscale('log')
ax8.set_xscale('log')
ax1.set_title('{0}    DM Power Spectrum'.format(dataSet),  fontsize=18)
ax3.set_title('Gas Power Spectrum',  fontsize=18)
ax5.set_title('Neutral Hydrogen Power Spectrum',  fontsize=18)
ax7.set_title('Ionized Hydrogen Power Spectrum',  fontsize=18)

# ax1.xlim()
fig.savefig( fileName,  pad_inches=0.1,  bbox_inches='tight', dpi=80)
print 'Saved Image: ', fileName
