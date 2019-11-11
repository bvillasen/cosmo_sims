import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
from power_spectrum import get_power_spectrum
import matplotlib

dev_dir = '/home/bruno/Desktop/Dropbox/Developer/'
cosmo_dir = dev_dir + 'cosmo_sims/'
toolsDirectory = cosmo_dir + "tools/"
sys.path.extend([toolsDirectory ] )
from load_data_cholla import load_snapshot_data, load_snapshot_data_particles
from load_data_ramses import load_snapshot_ramses
from load_data_nyx import load_snapshot_nyx
from mpi4py import MPI

# set some global options
matplotlib.font_manager.findSystemFonts(fontpaths=['/home/bruno/Downloads'], fontext='ttf')
# plt.rcParams['figure.figsize'] = (6,5)
# plt.rcParams['legend.frameon'] = False
# plt.rcParams['legend.fontsize'] = 14
# plt.rcParams['legend.borderpad'] = 0.1
# plt.rcParams['legend.labelspacing'] = 0.1
# plt.rcParams['legend.handletextpad'] = 0.1
plt.rcParams['font.family'] = 'Helvetica'
hfont = {'fontname':'Helvetica'}

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

dataDir = '/home/bruno/Desktop/data/'
# dataDir = '/raid/bruno/data/'


nPoints = 256
Lbox = 50.0   #Mpc/h

nyxDir = dataDir + 'cosmo_sims/nyx/256_dm_50Mpc/'
ramsesDir = dataDir + 'cosmo_sims/ramses/{0}_dm_50Mpc/h5_files/'.format(nPoints )
chollaDir_ramses = dataDir + 'cosmo_sims/cholla_pm/{0}_dm_50Mpc/data_ramses_2/'.format( nPoints )
chollaDir_nyx = dataDir + 'cosmo_sims/cholla_pm/{0}_dm_50Mpc/data_nyx/'.format( nPoints )
outDir = dev_dir + 'figures/power_dm/'
outputsDir = '/home/bruno/Desktop/Dropbox/Developer/cholla/scale_output_files/'

# fileName = outDir + 'ps_{0}_hydro_ramses_{2}_eta{1:.3f}.png'.format(  nPoints, eta, reconst )

# set simulation volume dimentions
nz, ny, nx = nPoints, nPoints, nPoints
nCells  = nx*ny*nz
h = 0.6766
Lx = Lbox
Ly = Lbox
Lz = Lbox
dx, dy, dz = Lx/(nx), Ly/(ny), Lz/(nz )

n_kSamples = 20
snapshots_ramses = [ 0, 3, 4, 6, 7, 8, 9, 10, 11,  14]

outputs_ramses = np.loadtxt( outputsDir + 'outputs_dm_ramses_256_50Mpc.txt')
outputs_nyx = np.loadtxt( outputsDir + 'outputs_dm_nyx_256_50Mpc.txt')
z_ramses = 1./(outputs_ramses) - 1
z_nyx = 1./(outputs_nyx) - 1

snapshots_nyx = []
for z in z_ramses[snapshots_ramses]:  
  z_diff_nyx = np.abs( z_nyx - z )
  index_nyx = np.where( z_diff_nyx == z_diff_nyx.min())[0][0]
  snapshots_nyx.append( index_nyx )

snapshots = snapshots_ramses

n_snapshots = len(snapshots)

if rank >= n_snapshots: exit()
nSnap = snapshots[rank]

n_power_data = 4
ps_all = np.ones( [n_power_data, n_kSamples] ) 
# ps_all *= rank

print " Cholla: ", nSnap

nSnap = snapshots_ramses[rank]
data_cholla = load_snapshot_data_particles( nSnap, chollaDir_ramses, single_file=True )
current_z_ch_ramses = data_cholla['current_z'][0]
dens_dm_cholla = data_cholla['density'][...]
ps_dm_cholla_ramses, k_vals, count_dm_cholla = get_power_spectrum( dens_dm_cholla, Lbox, nx, ny, nz, dx, dy, dz,  n_kSamples=n_kSamples)

print ' Ramses: ', nSnap
data_ramses = load_snapshot_ramses( nSnap, ramsesDir, dm=True, hydro=False, cool=False, particles=False)
current_a_ramses = data_ramses['current_a']
current_z_ramses = data_ramses['current_z']
dens_dm_ramses = data_ramses['dm']['density'][...]
ps_dm_ramses, k_vals, count_dm_ramses = get_power_spectrum( dens_dm_ramses, Lbox, nx, ny, nz, dx, dy, dz,  n_kSamples=n_kSamples)

nSnap = snapshots_nyx[rank]
data_cholla = load_snapshot_data_particles( nSnap, chollaDir_nyx, )
current_z_ch_nyx = data_cholla['current_z']
dens_dm_cholla = data_cholla['density'][...]
ps_dm_cholla_nyx, k_vals, count_dm_cholla = get_power_spectrum( dens_dm_cholla, Lbox, nx, ny, nz, dx, dy, dz,  n_kSamples=n_kSamples)



data_nyx = load_snapshot_nyx( nSnap, nyxDir, hydro=False, particles=False )
current_a_nyx = data_nyx['dm']['current_a']
current_z_nyx = data_nyx['dm']['current_z']
print current_z_nyx
dens_dm_nyx = data_nyx['dm']['density'][...]
ps_dm_nyx, k_vals, count_dm_ramses = get_power_spectrum( dens_dm_nyx, Lbox, nx, ny, nz, dx, dy, dz,  n_kSamples=n_kSamples)



ps_all[0] = ps_dm_cholla_ramses
ps_all[1] = ps_dm_cholla_nyx
ps_all[2] = ps_dm_ramses
ps_all[3] = ps_dm_nyx


send_buf = ps_all
recv_buf = None
if rank == 0:
  recv_buf = np.empty ([ n_snapshots, n_power_data, n_kSamples], dtype=np.float64)
comm.Gather(send_buf, recv_buf, root=0)
data_all = recv_buf

send_buf = np.array([current_z_ch_nyx])
recv_buf = None
if rank == 0:
  recv_buf = np.empty ([ n_snapshots ], dtype=np.float64)
comm.Gather(send_buf, recv_buf, root=0)
current_z_all_nyx = recv_buf


send_buf = np.array([current_z_ch_ramses])
recv_buf = None
if rank == 0:
  recv_buf = np.empty ([ n_snapshots ], dtype=np.float64)
comm.Gather(send_buf, recv_buf, root=0)
current_z_all_ramses = recv_buf

if rank != 0: exit()


# print data_all
# print current_z_all

fig = plt.figure(0)
fig.set_size_inches(16,10)
fig.clf()

gs = plt.GridSpec(5, 2)
gs.update(hspace=0.06, wspace=0.13, )
ax1 = plt.subplot(gs[0:4, 0])
ax2 = plt.subplot(gs[4:5, 0])

ax3 = plt.subplot(gs[0:4, 1])
ax4 = plt.subplot(gs[4:5, 1])

# colors = ['b', 'y', 'g', 'c', 'm', 'b', 'y', 'g', 'c', 'm', ]
colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

for i in range(n_snapshots):

  i = n_snapshots-1 - i

  ps_dm_cholla_ramses = data_all[i,0]
  ps_dm_cholla_nyx = data_all[i,1]
  ps_dm_ramses = data_all[i,2]
  ps_dm_nyx = data_all[i,3]



  label_nyx = 'z = {0:.1f}'.format(np.abs(current_z_all_nyx[i]))
  label_ramses = 'z = {0:.1f}'.format(np.abs(current_z_all_ramses[i]))
  c = colors[i]
  if i == n_snapshots-1:
    ax1.plot( k_vals, ps_dm_nyx, '--', c='k', linewidth=1, label='Nyx' )
    ax3.plot( k_vals, ps_dm_ramses, '--', c='k', linewidth=1, label='Ramses' )
    
  ax1.plot( k_vals, ps_dm_cholla_nyx, c=c,  linewidth=3, label=label_nyx )
  ax3.plot( k_vals, ps_dm_cholla_ramses, c=c,  linewidth=3, label=label_ramses )

  ax1.plot( k_vals, ps_dm_nyx, '--', c='k', linewidth=1 )
  ax3.plot( k_vals, ps_dm_ramses, '--', c='k', linewidth=1 )
  #
  error_dm_nyx = (ps_dm_cholla_nyx - ps_dm_nyx) / ps_dm_nyx
  error_dm_ramses = (ps_dm_cholla_ramses - ps_dm_ramses) / ps_dm_ramses
  # error_gas = (ps_gas_cholla - ps_gas_ramses) / ps_gas_cholla

  ax2.plot( k_vals, error_dm_nyx , c=c, alpha=0.9)
  ax4.plot( k_vals, error_dm_ramses , c=c, alpha=0.9)

ax2.axhline( y=0., color='r', linestyle='--',  )
ax2.set_ylim( -0.004, 0.004)
ax4.axhline( y=0., color='r', linestyle='--',  )
ax4.set_ylim( -0.02, 0.02)
ax2.ticklabel_format(axis='both', style='sci')

ax1.tick_params(axis='both', which='major', labelsize=13, size=5)
ax1.tick_params(axis='both', which='minor', labelsize=10, size=3)
ax2.tick_params(axis='both', which='major', labelsize=13, size=5)
ax2.tick_params(axis='both', which='minor', labelsize=10, size=3)

ax3.tick_params(axis='both', which='major', labelsize=13, size=5)
ax3.tick_params(axis='both', which='minor', labelsize=10, size=3)
ax4.tick_params(axis='both', which='major', labelsize=13, size=5)
ax4.tick_params(axis='both', which='minor', labelsize=10, size=3)

ax1.text(0.96, 0.93, 'Dark Matter Power Spectrum\nComparison to Nyx', fontsize=17, horizontalalignment='right', verticalalignment='center', transform=ax1.transAxes, **hfont)
ax3.text(0.96, 0.93, 'Dark Matter Power Spectrum\nComparison to Ramses', fontsize=17, horizontalalignment='right', verticalalignment='center', transform=ax3.transAxes, **hfont)


ax1.set_ylabel( r'$P(k)$   $[h^3 \mathrm{Mpc}^{-3}]$', fontsize=17)
ax2.set_ylabel( 'Fractional Difference', fontsize=14)
ax1.legend( loc=3, fontsize=12)
ax2.set_xlabel( r'$k \, \, \, \,[h \mathrm{Mpc}^{-1}]$', fontsize=17)

ax3.legend( loc=3, fontsize=12)
ax4.set_xlabel( r'$k \, \, \,\,[h \mathrm{Mpc}^{-1}]$', fontsize=17)

ax1.set_xscale('log')
ax1.set_yscale('log')
ax2.set_xscale('log')
ax3.set_xscale('log')
ax3.set_yscale('log')
ax4.set_xscale('log')
# ax1.set_title('DM Power Spectrum',  fontsize=18)
# ax3.set_title('Gas Power Spectrum ',  fontsize=18)


# fig.suptitle(r'{0} '.format(data_name ), fontsize=20, y=0.95)

# ax1.xlim()
fileName = outDir + 'ps_{0}_dm_nyx_ramses.png'.format( nPoints )

fig.savefig( fileName,  pad_inches=0.1,  bbox_inches='tight', dpi=300)
print 'Saved Image: ', fileName











