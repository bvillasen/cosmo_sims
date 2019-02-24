import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
from power_spectrum import get_power_spectrum

cosmo_dir = '/home/bruno/Desktop/Dropbox/Developer/cosmo_sims/'
toolsDirectory = cosmo_dir + "tools/"
sys.path.extend([toolsDirectory ] )
from load_data_cholla import load_snapshot_data


dataDir = '/raid/bruno/data/'
chollaDir_hm = dataDir + 'cosmo_sims/cholla_pm/512_cool/data_uv_HM/'

outDir = cosmo_dir + 'figures/power_hydro/'


# set simulation volume dimentions
nPoints = 512
nz, ny, nx = nPoints, nPoints, nPoints
nCells  = nx*ny*nz
Lbox = 50.0   #Mpc/h
h = 0.6774
Lx = Lbox
Ly = Lbox
Lz = Lbox
dx, dy, dz = Lx/(nx), Ly/(ny), Lz/(nz )

n_kSamples = 20


fileName = outDir + 'ps_512.png'

snapshots = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# snapshots = [ 0]
fig = plt.figure(0)
fig.set_size_inches(30,10)
fig.clf()


gs = plt.GridSpec(5, 3)
gs.update(hspace=0.05, wspace=0.08, )
ax1 = plt.subplot(gs[0:4, 0])
ax2 = plt.subplot(gs[4:5, 0])

ax3 = plt.subplot(gs[0:4, 1])
ax4 = plt.subplot(gs[4:5, 1])

ax5 = plt.subplot(gs[0:4, 2])
ax6 = plt.subplot(gs[4:5, 2])

colors = ['b', 'y', 'g', 'c', 'm', 'b', 'y', 'g', 'c', 'm', 'b' ]

for i,nSnap in enumerate(snapshots):
  print " Cholla: ", nSnap
  snapKey = str( nSnap )
  # if i not in [9]: continue
  data_cholla = load_snapshot_data( snapKey, chollaDir_hm, cool=True )
  current_z = data_cholla['current_z']
  dens_dm = data_cholla['dm']['density'][...]
  dens_gas = data_cholla['gas']['density'][...]
  dens_gas_H = data_cholla['gas']['HI_density'][...]

  ps_dm_hm, k_vals, count_dm_cholla = get_power_spectrum( dens_dm, Lbox, nx, ny, nz, dx, dy, dz,  n_kSamples=n_kSamples)
  ps_gas_hm, k_vals, count_gas_cholla = get_power_spectrum( dens_gas, Lbox, nx, ny, nz, dx, dy, dz,  n_kSamples=n_kSamples)
  ps_gas_H_hm, k_vals, count_gas_cholla = get_power_spectrum( dens_gas_H, Lbox, nx, ny, nz, dx, dy, dz,  n_kSamples=n_kSamples)

  c= colors[i]
  label = 'z = {0:.1f}_HM'.format(current_z)
  ax1.plot( k_vals, ps_dm_hm,  c=c, linewidth=2, label=label )
  ax3.plot( k_vals, ps_gas_hm, c=c,  linewidth=2, label=label )
  ax5.plot( k_vals, ps_gas_H_hm, c=c,  linewidth=2, label=label )

  data_cholla = load_snapshot_data( snapKey, chollaDir_fg, cool=True )
  current_z = data_cholla['current_z']
  dens_dm = data_cholla['dm']['density'][...]
  dens_gas = data_cholla['gas']['density'][...]
  dens_gas_H = data_cholla['gas']['HI_density'][...]

  ps_dm_fg, k_vals, count_dm_cholla = get_power_spectrum( dens_dm, Lbox, nx, ny, nz, dx, dy, dz,  n_kSamples=n_kSamples)
  ps_gas_fg, k_vals, count_gas_cholla = get_power_spectrum( dens_gas, Lbox, nx, ny, nz, dx, dy, dz,  n_kSamples=n_kSamples)
  ps_gas_H_fg, k_vals, count_gas_cholla = get_power_spectrum( dens_gas_H, Lbox, nx, ny, nz, dx, dy, dz,  n_kSamples=n_kSamples)
#
  # # label = 'z = {0:.1f}'.format(current_z_enzo)
  if i == 9:
    ax1.plot( k_vals, ps_dm_fg, '--', c='k', linewidth=1, label='Enzo' )
    ax3.plot( k_vals, ps_gas_fg, '--', c='k', linewidth=1, label='Enzo' )
    ax5.plot( k_vals, ps_gas_H_fg, '--', c='k', linewidth=1 )
  else:
    ax1.plot( k_vals, ps_dm_fg, '--', c='k', linewidth=1 )
    ax3.plot( k_vals, ps_gas_fg, '--', c='k', linewidth=1 )
    ax5.plot( k_vals, ps_gas_H_fg, '--', c='k', linewidth=1 )
  #
  #
  error_dm = (ps_dm_hm - ps_dm_fg) / ps_dm_hm
  error_gas = (ps_gas_hm - ps_gas_fg) / ps_gas_hm
  error_gas_H = (ps_gas_H_hm - ps_gas_H_fg) / ps_gas_H_hm

  ax2.plot( k_vals, error_dm , c=c, alpha=0.9)
  ax4.plot( k_vals, error_gas , c=c, alpha=0.9)
  ax4.plot( k_vals, error_gas_H , c=c, alpha=0.9)

ax2.axhline( y=0., color='r', linestyle='--',  )
ax2.set_ylim( -0.4, 0.4)
ax4.axhline( y=0., color='r', linestyle='--',  )
ax4.set_ylim( -0.4, 0.4)
ax6.axhline( y=0., color='r', linestyle='--',  )
ax6.set_ylim( -0.4, 0.4)

ax1.set_ylabel( r'$P(k) $', fontsize=17)
ax2.set_ylabel( 'Error', fontsize=15)
ax1.legend( loc=3)
ax2.set_xlabel( r'$k \, \, [h Mpc^{-1}]$', fontsize=17)
ax3.legend( loc=3)
ax4.set_xlabel( r'$k \, \, [h Mpc^{-1}]$', fontsize=17)
ax5.legend( loc=3)
ax6.set_xlabel( r'$k \, \, [h Mpc^{-1}]$', fontsize=17)

ax1.set_xscale('log')
ax1.set_yscale('log')
ax3.set_xscale('log')
ax3.set_yscale('log')
ax5.set_xscale('log')
ax5.set_yscale('log')
ax2.set_xscale('log')
ax4.set_xscale('log')
ax6.set_xscale('log')


ax1.set_title('DM Power Spectrum',  fontsize=18)
ax3.set_title('Gas Power Spectrum',  fontsize=18)
ax5.set_title('Neutral Hydrogen Power Spectrum',  fontsize=18)
# ax1.xlim()
fig.savefig( fileName,  pad_inches=0.1,  bbox_inches='tight', dpi=80)
print 'Saved Image: ', fileName
