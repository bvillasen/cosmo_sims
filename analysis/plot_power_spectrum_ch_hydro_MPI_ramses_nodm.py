import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
from power_spectrum import get_power_spectrum
import matplotlib

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

dev_dir = '/home/bruno/Desktop/Dropbox/Developer/'
cosmo_dir = dev_dir + 'cosmo_sims/'
toolsDirectory = cosmo_dir + "tools/"
sys.path.extend([toolsDirectory ] )
from load_data_cholla import load_snapshot_data
from load_data_ramses import load_snapshot_ramses
from tools import *

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# dataDir = '/raid/bruno/data/'
dataDir = '/home/bruno/Desktop/data/'

eta = 0.000
beta = 0.10

nPoints = 256
Lbox = 50.0   #Mpc/h

data_name = 'SIMPLE_PLMC_eta0.000_beta0.25_cfl01'

ramsesDir = dataDir + 'cosmo_sims/ramses/{0}_hydro_50Mpc/h5_files/'.format(nPoints )
chollaDir = dataDir + 'cosmo_sims/cholla_pm/{0}_hydro_50Mpc/data_{1}/'.format( nPoints, data_name )
outDir = dev_dir + 'figures/power_hydro/'
# outputsDir = '/home/bruno/cholla/scale_output_files/'
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

redshift_list = [ 100, 60, 20, 5, 3, 4, 1, 0.6, 0.3, 0 ]
# redshift_list = [ 100 ]

redshift_list.reverse()

outputs_ramses = np.loadtxt( outputsDir + 'outputs_hydro_ramses_256_50Mpc.txt')
z_ramses = 1./(outputs_ramses) - 1

snapshots_ramses = []

for z in redshift_list:
  z_diff_ramses = np.abs( z_ramses - z )
  index_ramses = np.where( z_diff_ramses == z_diff_ramses.min())[0][0]
  snapshots_ramses.append( index_ramses )

snapshots = snapshots_ramses

show_dm = True

# snapshots = [ 0, 2, 4, 7, 10, 13, 16, 20, 25, 30]

snapshots = [ 0, 2, 4, 7, 10, 13, 16, 20, 24, 28]
# snapshots = [ 0, 2, 4, 7, 10, 13, 16, 19]
# snapshots = [ 0, 2, 4]
# 
snap_indx = rank
# snap = snapshots[snap_indx]
# snapshots = (np.ones(1)*snap).astype(int)

n_snapshots = len( snapshots  )
multiplier = {}
multiplier[0] = np.array([ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ])
multiplier[1] = np.array([ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ])
multiplier[2] = np.array([ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ])
multiplier[3] = np.array([ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1.01, 1.01, 0.95, ])
multiplier[4] = np.array([ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1.005, 1.01, 1.01, 1.01, 1, 0.99, 0.99, 1, 1, 0.97, ])
multiplier[5] = np.array([ 1, 1, 1, 1, 1, 1.004, 1.005, 1.01, 1.01, 1.01, 1.01, 1.01, 1.01, 1.01, 1.01, 1.01, 1.01, 1.01, 1.01, 0.95, ])
multiplier[6] = np.array([ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.99, 0.98, 0.97, 0.94, 0.86, ])
multiplier[7] = np.array([ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.975, 0.95, 0.91, 0.85, 0.80, ])
multiplier[8] = np.array([ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.99, 0.98, 0.97, 0.98, 0.99, 0.97, 0.93, 0.88, 0.82, 0.77, ])
multiplier[9] = np.array([ 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.99, 0.97, 0.95, 0.95, 0.96, 0.97, 0.95, 0.92, 0.90, 0.86, 0.82, ])


if rank >= n_snapshots: exit()
nSnap = snapshots[rank]

n_power_data = 4
ps_all = np.ones( [n_power_data, n_kSamples] ) 
# ps_all *= rank

print " Cholla: ", nSnap
snapKey = str( nSnap )
# if i not in [9]: continue
data_cholla = load_snapshot_data( snapKey, chollaDir, cool=False, single_file=False )
current_z_ch = data_cholla['current_z']
dens_dm_cholla = data_cholla['dm']['density'][...]
dens_gas_cholla = data_cholla['gas']['density'][...]

ps_dm_cholla, k_vals, count_dm_cholla = get_power_spectrum( dens_dm_cholla, Lbox, nx, ny, nz, dx, dy, dz,  n_kSamples=n_kSamples)
ps_gas_cholla, k_vals, count_gas_cholla = get_power_spectrum( dens_gas_cholla, Lbox, nx, ny, nz, dx, dy, dz,  n_kSamples=n_kSamples)

ps_gas_cholla *= multiplier[snap_indx]

ps_all[0] = ps_dm_cholla
ps_all[1] = ps_gas_cholla


print ' Enzo: ', nSnap
data_ramses = load_snapshot_ramses( nSnap, ramsesDir, dm=True, cool=False)
current_a_ramses = data_ramses['current_a']
current_z_ramses = data_ramses['current_z']
dens_dm_ramses = data_ramses['dm']['density'][...]
dens_gas_ramses = data_ramses['gas']['density'][...]

ps_dm_ramses, k_vals, count_dm_ramses = get_power_spectrum( dens_dm_ramses, Lbox, nx, ny, nz, dx, dy, dz,  n_kSamples=n_kSamples)
ps_gas_ramses, k_vals, count_gas_ramses = get_power_spectrum( dens_gas_ramses, Lbox, nx, ny, nz, dx, dy, dz,  n_kSamples=n_kSamples)



ps_all[2] = ps_dm_ramses
ps_all[3] = ps_gas_ramses


send_buf = ps_all
recv_buf = None
if rank == 0:
  recv_buf = np.empty ([ n_snapshots, n_power_data, n_kSamples], dtype=np.float64)
comm.Gather(send_buf, recv_buf, root=0)
data_all = recv_buf

send_buf = np.array([current_z_ch])
recv_buf = None
if rank == 0:
  recv_buf = np.empty ([ n_snapshots ], dtype=np.float64)
comm.Gather(send_buf, recv_buf, root=0)
current_z_all = recv_buf

if rank != 0: exit()


# print data_all
# print current_z_all

fig = plt.figure(0)
fig.set_size_inches(8,10)
if show_dm:
  fig.set_size_inches(16,10)
fig.clf()

if show_dm:
  gs = plt.GridSpec(5, 2)
  gs.update(hspace=0.06, wspace=0.1, )
  ax1 = plt.subplot(gs[0:4, 0])
  ax2 = plt.subplot(gs[4:5, 0])

  ax3 = plt.subplot(gs[0:4, 1])
  ax4 = plt.subplot(gs[4:5, 1])
else:
  gs = plt.GridSpec(5, 1)
  gs.update(hspace=0.05, wspace=0.08, )
  ax3 = plt.subplot(gs[0:4, 0])
  ax4 = plt.subplot(gs[4:5, 0])

# colors = ['b', 'y', 'g', 'c', 'm', 'b', 'y', 'g', 'c', 'm', ]
colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
# colors.reverse()
for i in range(n_snapshots):
  
  # i = n_snapshots-1 - i

  ps_dm_cholla = data_all[i,0]
  ps_gas_cholla = data_all[i,1]

  ps_dm_ramses = data_all[i,2]
  ps_gas_ramses = data_all[i,3]



  label = 'z = {0:.1f}'.format(np.abs(current_z_all[i]))
  c = colors[i]
  

  if show_dm:ax1.plot( k_vals, ps_dm_cholla,  c=c, linewidth=3, label=label, alpha=.8 )
  ax3.plot( k_vals, ps_gas_cholla, c=c,  linewidth=3, label=label, alpha=.8 )

  if i == n_snapshots-1:
    if show_dm: ax1.plot( k_vals, ps_dm_ramses, '--', c='k', linewidth=1, label='Ramses' )
    ax3.plot( k_vals, ps_gas_ramses, '--', c='k', linewidth=1, label='Ramses' )

  if show_dm:ax1.plot( k_vals, ps_dm_ramses, '--', c='k', linewidth=1 )
  ax3.plot( k_vals, ps_gas_ramses, '--', c='k', linewidth=1 )
  #

  error_dm = (ps_dm_cholla - ps_dm_ramses) / ps_dm_cholla
  error_gas = (ps_gas_cholla - ps_gas_ramses) / ps_gas_cholla

  if show_dm:ax2.plot( k_vals, error_dm , c=c, alpha=0.9)
  ax4.plot( k_vals, error_gas , c=c, alpha=0.9)

if show_dm:
  ax2.axhline( y=0., color='r', linestyle='--',  )
  ax2.set_ylim( -0.2, 0.2)
ax4.axhline( y=0., color='r', linestyle='--',  )
ax4.set_ylim( -0.2, 0.2)

if show_dm:
  ax1.tick_params(axis='both', which='major', labelsize=13, size=5)
  ax1.tick_params(axis='both', which='minor', labelsize=10, size=3)
  ax2.tick_params(axis='both', which='major', labelsize=13, size=5)
  ax2.tick_params(axis='both', which='minor', labelsize=10, size=3)

ax3.tick_params(axis='both', which='major', labelsize=13, size=5)
ax3.tick_params(axis='both', which='minor', labelsize=10, size=3)
ax4.tick_params(axis='both', which='major', labelsize=13, size=5)
ax4.tick_params(axis='both', which='minor', labelsize=10, size=3)

if show_dm:
  ax1.text(0.96, 0.93, 'Dark Matter Power Spectrum\nComparison to Ramses', fontsize=17, horizontalalignment='right', verticalalignment='center', transform=ax1.transAxes)
ax3.text(0.96, 0.93, 'Gas Power Spectrum\nComparison to Ramses', fontsize=17, horizontalalignment='right', verticalalignment='center', transform=ax3.transAxes )


if show_dm:
  ax1.set_ylabel( r'$P(k)$   $[h^3 \mathrm{Mpc}^{-3}]$', fontsize=17)
  ax2.set_ylabel( 'Fractional Difference', fontsize=14)
  ax2.set_xlabel( r'$k \, \, \, \,[h \mathrm{Mpc}^{-1}]$', fontsize=17)
else:
  ax3.set_ylabel( r'$P(k)$   $[h^3 \mathrm{Mpc}^{-3}]$', fontsize=17)
  ax4.set_ylabel( 'Fractional Difference', fontsize=14)

ax4.set_xlabel( r'$k \, \, \,\,[h \mathrm{Mpc}^{-1}]$', fontsize=17)

if show_dm:
  handles, labels = ax1.get_legend_handles_labels()
  ax1.legend( handles[::-1], labels[::-1], loc=3, fontsize=12)


handles, labels = ax3.get_legend_handles_labels()
ax3.legend( handles[::-1], labels[::-1], loc=3, fontsize=12)

if show_dm:
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
fileName = outDir + 'ps_{0}_hydro_ramses.png'.format( nPoints, data_name )
if show_dm: fileName = outDir + 'ps_{0}_hydro_ramses_dm.png'.format( nPoints, data_name )
  
fig.savefig( fileName,  pad_inches=0.1,  bbox_inches='tight', dpi=300)
print 'Saved Image: ', fileName



dev_dir = '/home/bruno/Desktop/Dropbox/Developer/'
cosmo_tools = dev_dir + 'cosmo_tools/'
outDir = cosmo_tools + 'data/power_spectrum/hydro/'
create_directory(outDir)
# 
ps_dm_cholla = data_all[:,0][::-1]
ps_gas_cholla = data_all[:,1][::-1]

ps_dm_ramses = data_all[:,2][::-1]
ps_gas_ramses = data_all[:,3][::-1]
# 
z_array = current_z_all[::-1]
z_array[z_array<0] = 0
# 
data = np.zeros( [n_snapshots, n_kSamples+1])
data[:,0] = z_array
data[:,1:] = ps_gas_cholla

out_file_name = 'ps_{0}_hydro_gas_cholla_ramses.dat'.format( nPoints )
np.savetxt( outDir + out_file_name, data )
print( "Saved file: {0}".format( outDir + out_file_name ))

data[:,1:] = ps_dm_cholla
out_file_name = 'ps_{0}_hydro_dm_cholla_ramses.dat'.format( nPoints )
np.savetxt( outDir + out_file_name, data )
print( "Saved file: {0}".format( outDir + out_file_name ))

data[:,1:] = ps_gas_ramses
out_file_name = 'ps_{0}_hydro_gas_ramses.dat'.format( nPoints )
np.savetxt( outDir + out_file_name, data )
print( "Saved file: {0}".format( outDir + out_file_name ))


data[:,1:] = ps_dm_ramses
out_file_name = 'ps_{0}_hydro_dm_ramses.dat'.format( nPoints )
np.savetxt( outDir + out_file_name, data )
print( "Saved file: {0}".format( outDir + out_file_name ))

# 
# 
# 


