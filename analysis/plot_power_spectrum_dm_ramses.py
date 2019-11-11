import sys
import numpy as np
import matplotlib.pyplot as plt
# from np.fft import fftn
# from colossus.cosmology import cosmology
from scipy.interpolate import RegularGridInterpolator
import h5py as h5
from power_spectrum import get_power_spectrum

# dataDir = '/raid/bruno/data/'
dataDir = '/home/bruno/Desktop/data/'
dev_dir = '/home/bruno/Desktop/Dropbox/Developer/'
cosmo_dir = dev_dir + 'cosmo_sims/'
toolsDirectory = cosmo_dir + "tools/"
sys.path.extend([toolsDirectory ] )
from load_data_cholla import load_snapshot_data_particles
from load_data_nyx import load_snapshot_nyx
from load_data_enzo import load_snapshot_enzo
from load_data_ramses import load_snapshot_ramses


nPoints = 256

extra_name = '_2'

outputsDir = '/home/bruno/cholla/scale_output_files/'
chollaDir_ramses = dataDir + 'cosmo_sims/cholla_pm/{0}_dm_50Mpc/data_ramses{1}/'.format( nPoints, extra_name )
ramsesDir = dataDir + 'cosmo_sims/ramses/{0}_dm_50Mpc/h5_files/'.format( nPoints )
outDir = dev_dir + 'figures/power_dm/'

fileName = 'power_dm_{0}_ramses{1}.png'.format( nPoints, extra_name )
# redshift_list = [ 100, 70, 40, 10, 7, 4, 1, 0.6, 0.3, 0 ]
# # redshift_list = [ 100, 70, 40, 10, 7, 4, 1, 0.6  ]
# redshift_list.reverse()
# 
# outputs_ramses = np.loadtxt( outputsDir + 'outputs_dm_ramses_256_50Mpc.txt')
# # outputs_nyx = np.loadtxt( outputsDir + 'outputs_dm_nyx_256_50Mpc.txt')
# z_ramses = 1./(outputs_ramses) - 1
# # z_nyx = 1./(outputs_nyx) - 1
# 
# 
# snapshots_ramses = []
# # snapshots_nyx = []
# # 
# for z in redshift_list:
#   z_diff_ramses = np.abs( z_ramses - z )
#   index_ramses = np.where( z_diff_ramses == z_diff_ramses.min())[0][0]
#   snapshots_ramses.append( index_ramses )
# 
# # for z in z_ramses[snapshots_ramses]:  
# #   z_diff_nyx = np.abs( z_nyx - z )
# #   index_nyx = np.where( z_diff_nyx == z_diff_nyx.min())[0][0]
# #   snapshots_nyx.append( index_nyx )
# 

snapshots = [0, 2, 3, 5, 6, 8, 9,  10, 11, 14]

# 
# Lbox = 115.0   #Mpc/h
# h = 0.6774

Lbox = 50.
h = 0.6766

nz, ny, nx = nPoints, nPoints, nPoints
nCells  = nx*ny*nz
Lx = Lbox
Ly = Lbox
Lz = Lbox
dx, dy, dz = Lx/(nx), Ly/(ny), Lz/(nz )
n_kSamples = 12


fig = plt.figure(0)
fig.set_size_inches(10,12)
fig.clf()


colors = [ 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11' ]

gs = plt.GridSpec(5, 1)
gs.update(hspace=0.05)
gs.update(wspace=0.1)
ax1 = plt.subplot(gs[0:4, 0])
ax2 = plt.subplot(gs[4:5, 0])
ax2.axhline( y=0., color='r', linestyle='--',  )


snapshots.reverse()
for i,nSnap in enumerate(snapshots):

  data_cholla = load_snapshot_data_particles( nSnap, chollaDir_ramses, single_file=True )
  current_z_ch_ramses = data_cholla['current_z']
  dens_dm_cholla_ramses = data_cholla['density'][...]
  print " Cholla: ", current_z_ch_ramses, dens_dm_cholla_ramses.mean()
  ps_dm_cholla_ramses, k_vals, error_cholla = get_power_spectrum( dens_dm_cholla_ramses, Lbox, nx, ny, nz, dx, dy, dz,  n_kSamples=n_kSamples)
  
  #Load ramses data
  data_ramses = load_snapshot_ramses( nSnap, ramsesDir, dm=True, particles=False, cool=False, metals=False, hydro=False)
  current_a_ramses = data_ramses['current_a']
  current_z_ramses = data_ramses['current_z']
  dens_dm_ramses = data_ramses['dm']['density'][...]
  print ' ramses: ', current_z_ramses, dens_dm_ramses.mean()
  ps_dm_ramses, k_vals, count_dm_ramses = get_power_spectrum( dens_dm_ramses, Lbox, nx, ny, nz, dx, dy, dz,  n_kSamples=n_kSamples)
  
  error_ramses = ( ps_dm_cholla_ramses - ps_dm_ramses ) /ps_dm_ramses
  print "Error: ", error_ramses.min(), error_ramses.max()


  print "Plotting..."
  c = colors[i]

  label = 'z = {0:.1f}'.format(np.abs(current_z_ch_ramses))
  if i==0: ax1.plot( k_vals, ps_dm_ramses, '--', c='k', label='Ramses' )
  ax1.plot( k_vals, ps_dm_cholla_ramses, label=label, c=c, linewidth=4 )
  ax1.plot( k_vals, ps_dm_ramses, '--', c='k' )
  ax2.plot( k_vals, error_ramses, c=c, alpha=0.9, linewidth=2 )





max_diff_ramses = 0.05
ax2.set_ylim( -1*max_diff_ramses, 1*max_diff_ramses )

ax1.set_ylabel( r'$P(k) \, \, \,  [h^3  \, Mpc^{-3}]$', fontsize=17)
ax2.set_ylabel( 'Difference', fontsize=15)
ax2.set_xlabel( r'$k \, \, [h  \, Mpc^{-1}]$', fontsize=17)

ax1.legend( loc=3, fontsize=15)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax2.set_xscale('log')


# plt.xlim()
ax1.set_title( r'DM Power Spectrum RAMSES - CHOLLA ', fontsize=20 )

print "Saved File: ", fileName
fig.savefig( outDir + fileName,  pad_inches=0.1,  bbox_inches='tight', dpi=300)