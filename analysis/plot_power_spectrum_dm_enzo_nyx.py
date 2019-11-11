import sys
import numpy as np
import matplotlib.pyplot as plt
# from np.fft import fftn
# from colossus.cosmology import cosmology
from scipy.interpolate import RegularGridInterpolator
import h5py as h5
from power_spectrum import get_power_spectrum

dataDir = '/raid/bruno/data/'
# dataDir = '/home/bruno/Desktop/hard_drive_1/data/'
dev_dir = '/home/bruno/Desktop/Dropbox/Developer/'
cosmo_dir = dev_dir + 'cosmo_sims/'
toolsDirectory = cosmo_dir + "tools/"
sys.path.extend([toolsDirectory ] )
from load_data_cholla import load_snapshot_data_particles
from load_data_nyx import load_snapshot_nyx
from load_data_enzo import load_snapshot_enzo


outputsDir = '/home/bruno/cholla/scale_output_files/'
nyxDir = dataDir + 'cosmo_sims/nyx/256_dm_50Mpc/'
enzoDir = dataDir + 'cosmo_sims/enzo/256_dm_50Mpc_grav4/h5_files/'
chollaDir_enzo = dataDir + 'cosmo_sims/cholla_pm/256_dm_50Mpc/data_enzo_grav4/'
chollaDir_nyx = dataDir + 'cosmo_sims/cholla_pm/256_dm_50Mpc/data_nyx_peculiar/'
outDir = dev_dir + 'figures/power_dm/'

fileName = 'power_dm_enzo_grav4.png'
redshift_list = [ 100, 70, 40, 10, 7, 4, 1, 0.6, 0.3, 0 ]
# redshift_list = [ 100, 70, 40, 10, 7, 4, 1, 0.6  ]
redshift_list.reverse()

outputs_enzo = np.loadtxt( outputsDir + 'outputs_dm_enzo_256_50Mpc_grav4.txt')
# outputs_nyx = np.loadtxt( outputsDir + 'outputs_dm_nyx_256_50Mpc.txt')
z_enzo = 1./(outputs_enzo) - 1
# z_nyx = 1./(outputs_nyx) - 1


snapshots_enzo = []
# snapshots_nyx = []
# 
for z in redshift_list:
  z_diff_enzo = np.abs( z_enzo - z )
  index_enzo = np.where( z_diff_enzo == z_diff_enzo.min())[0][0]
  snapshots_enzo.append( index_enzo )

# for z in z_enzo[snapshots_enzo]:  
#   z_diff_nyx = np.abs( z_nyx - z )
#   index_nyx = np.where( z_diff_nyx == z_diff_nyx.min())[0][0]
#   snapshots_nyx.append( index_nyx )



# 
# Lbox = 115.0   #Mpc/h
# h = 0.6774

Lbox = 50.
h = 0.6766

nPoints = 256
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




for i in range(len(redshift_list)):
  
  nSnap_enzo = snapshots_enzo[i]
  
  data_cholla = load_snapshot_data_particles( nSnap_enzo, chollaDir_enzo )
  current_z_ch_enzo = data_cholla['current_z']
  dens_dm_cholla_enzo = data_cholla['density'][...]
  print " Cholla: ", current_z_ch_enzo, dens_dm_cholla_enzo.mean()
  ps_dm_cholla_enzo, k_vals, error_cholla = get_power_spectrum( dens_dm_cholla_enzo, Lbox, nx, ny, nz, dx, dy, dz,  n_kSamples=n_kSamples)
  
  #Load Enzo data
  data_enzo = load_snapshot_enzo( nSnap_enzo, enzoDir, dm=True, particles=False, cool=False, metals=False, hydro=False)
  current_a_enzo = data_enzo['current_a']
  current_z_enzo = data_enzo['current_z']
  dens_dm_enzo = data_enzo['dm']['density'][...]
  print ' Enzo: ', current_z_enzo, dens_dm_enzo.mean()
  ps_dm_enzo, k_vals, count_dm_enzo = get_power_spectrum( dens_dm_enzo, Lbox, nx, ny, nz, dx, dy, dz,  n_kSamples=n_kSamples)
  
  error_enzo = ( ps_dm_cholla_enzo - ps_dm_enzo ) /ps_dm_enzo
  # print "Error: {0}\n".format(error.max())


  print "Plotting..."
  c = colors[i]

  label = 'z = {0:.1f}'.format(np.abs(current_z_ch_enzo))
  if i==0: ax1.plot( k_vals, ps_dm_enzo, '--', c='k', label='Enzo' )
  ax1.plot( k_vals, ps_dm_cholla_enzo, label=label, c=c, linewidth=4 )
  ax1.plot( k_vals, ps_dm_enzo, '--', c='k' )
  ax2.plot( k_vals, error_enzo, c=c, alpha=0.9, linewidth=2 )





max_diff_enzo = 0.5
ax2.set_ylim( -1*max_diff_enzo, 1*max_diff_enzo )

ax1.set_ylabel( r'$P(k) \, \, \,  [h^3  \, Mpc^{-3}]$', fontsize=17)
ax2.set_ylabel( 'Difference', fontsize=15)
ax2.set_xlabel( r'$k \, \, [h  \, Mpc^{-1}]$', fontsize=17)

ax1.legend( loc=3, fontsize=15)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax2.set_xscale('log')


# plt.xlim()
ax1.set_title( r'DM Power Spectrum ENZO - CHOLLA', fontsize=20 )

print "Saved File: ", fileName
fig.savefig( outDir + fileName,  pad_inches=0.1,  bbox_inches='tight', dpi=300)