import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# from np.fft import fftn
# from colossus.cosmology import cosmology
from scipy.interpolate import RegularGridInterpolator
import h5py as h5
from power_spectrum import get_power_spectrum

# # set some global options
# matplotlib.font_manager.findSystemFonts(fontpaths=['/home/bruno/Downloads'], fontext='ttf')
# # plt.rcParams['figure.figsize'] = (6,5)
# # plt.rcParams['legend.frameon'] = False
# # plt.rcParams['legend.fontsize'] = 14
# # plt.rcParams['legend.borderpad'] = 0.1
# # plt.rcParams['legend.labelspacing'] = 0.1
# # plt.rcParams['legend.handletextpad'] = 0.1
# plt.rcParams['font.family'] = 'Helvetica'
# # plt.rcParams['font.size'] = 16

dataDir = '/raid/bruno/data/'
# dataDir = '/home/bruno/Desktop/hard_drive_1/data/'
dev_dir = '/home/bruno/Desktop/Dropbox/Developer/'
cosmo_dir = dev_dir + 'cosmo_sims/'
toolsDirectory = cosmo_dir + "tools/"
sys.path.extend([toolsDirectory ] )
from load_data_cholla import load_snapshot_data_particles
from load_data_nyx import load_snapshot_nyx

nyxDir = dataDir + 'cosmo_sims/nyx/256_dm_50Mpc/'
chollaDir = dataDir + 'cosmo_sims/cholla_pm/256_dm_50Mpc/data_pfft_8/'
outDir = dev_dir + 'figures/power_dm/'
# outputsDir = '/home/bruno/Desktop/Dropbox/Developer/cholla/scale_output_files/'
outputsDir = '/home/bruno/cholla/scale_output_files/'

# snapshots_ch = [  2, 4, 6, 8, 10, 12, 16, 20, 24, 30 ]
# snapshots = range(0,300,35)
# snapshots.append(322)
# snapshots.append(94)

# snapshots = range(0,250,30)
# snapshots.append(258)
n_kSamples = 20

redshift_list = [ 100, 70, 40, 10, 7, 4, 1, 0.6, 0.3, 0 ]
# redshift_list = [ 100 ]

redshift_list.reverse()

outputs_nyx = np.loadtxt( outputsDir + 'outputs_dm_nyx_256_50Mpc.txt')
z_nyx = 1./(outputs_nyx) - 1

snapshots_nyx = []

for z in redshift_list:
  z_diff_nyx = np.abs( z_nyx - z )
  index_nyx = np.where( z_diff_nyx == z_diff_nyx.min())[0][0]
  snapshots_nyx.append( index_nyx )

snapshots = snapshots_nyx

print len(snapshots)
print snapshots

# snapshots = [0, 94]

fileName = 'power_dm_nyx_50Mpc_new_pfft_8.png'

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
n_kSamples = 20

fig = plt.figure(0)
fig.set_size_inches(10,12)
fig.clf()


colors = [ 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11' ]

gs = plt.GridSpec(5, 1)
gs.update(hspace=0.05)
ax1 = plt.subplot(gs[0:4, 0])
ax2 = plt.subplot(gs[4:5, 0])
ax2.axhline( y=0., color='r', linestyle='--',  )

# n_lines = len( snapshots)
# ax1.set_color_cycle([plt.cm.plasma_r(i) for i in np.linspace(0, 1, n_lines)])
# ax2.set_color_cycle([plt.cm.plasma_r(i) for i in np.linspace(0, 1, n_lines)])

for i,nSnap in enumerate(snapshots):

  # i = np.abs( i - len(snapshots) ) -1
  # nSnap = snapshots[i]

  print 'Snapshot: ', nSnap
  # if i not in [1,]: continue
  snapKey = str(nSnap)
  # c = colors[i]
  c = "C{0}".format(i)
  # Load cholla densities
  data_cholla = load_snapshot_data_particles( nSnap, chollaDir, single_file=False)
  current_z_ch_1 = data_cholla['current_z']
  dens_dm_cholla_1 = data_cholla['density'][...]
  print " Cholla: ", current_z_ch_1, dens_dm_cholla_1.mean()

  ps_dm_cholla, k_vals, error_cholla = get_power_spectrum( dens_dm_cholla_1, Lbox, nx, ny, nz, dx, dy, dz,  n_kSamples=n_kSamples)

  #Load Nyx data
  print ' Nyx: ', nSnap
  data_nyx = load_snapshot_nyx( nSnap, nyxDir, hydro=False, particles=False )
  current_a_nyx = data_nyx['dm']['current_a']
  current_z_nyx = data_nyx['dm']['current_z']
  print current_z_nyx
  dens_dm_nyx = data_nyx['dm']['density'][...]

  ps_dm_nyx, k_vals, count_dm_nyx = get_power_spectrum( dens_dm_nyx, Lbox, nx, ny, nz, dx, dy, dz,  n_kSamples=n_kSamples)

  error = ( ps_dm_cholla - ps_dm_nyx ) /ps_dm_cholla

  print "Error: {0}\n".format(error.max())


  print "Plotting..."
  if i==0: ax1.plot( k_vals, ps_dm_nyx, '--', c='k', label='Nyx' )
  label = 'z = {0:.1f}'.format(np.abs(current_z_ch_1))
  ax1.plot( k_vals, ps_dm_cholla, label=label, linewidth=4 )
  # plt.errorbar( k_vals, ps_dm_cholla, yerr=error_cholla, color='g')
  ax1.plot( k_vals, ps_dm_nyx, '--', c='k' )
  # plt.errorbar( k_vals, ps_dm_enzo, yerr=error_enzo, color='r')
  # plt.plot( k_vals, ps_dm_gad, '--', c='r', alpha=0.7 )
  # plt.plot( k_vals, power_spectrum * growthFactor**2, c='k' )

  ax2.plot( k_vals, error, alpha=0.8, linewidth=2 )


#
ax2.set_ylim( -0.005, 0.005 )

ax1.set_ylabel( r'$P(k) \, \, \,  [h^3  \, Mpc^{-3}]$', fontsize=17)
ax2.set_ylabel( 'Difference', fontsize=15)
ax2.set_xlabel( r'$k \, \, [h  \, Mpc^{-1}]$', fontname="Helvetica", fontsize=17, )
ax1.legend( loc=3)
ax1.set_xscale('log')
ax1.set_yscale('log')

ax2.set_xscale('log')
# plt.xlim()
# ax1.set_title( r'DM Power Spectrum   {0:.0f} Mpc/h  Box'.format(Lbox), fontsize=20)
print "Saved File: ", fileName
fig.savefig( outDir + fileName,  bbox_inches='tight', dpi=300)

































