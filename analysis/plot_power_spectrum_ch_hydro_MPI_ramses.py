import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
from power_spectrum import get_power_spectrum

dev_dir = '/home/bruno/Desktop/Dropbox/Developer/'
cosmo_dir = dev_dir + 'cosmo_sims/'
toolsDirectory = cosmo_dir + "tools/"
sys.path.extend([toolsDirectory ] )
from load_data_cholla import load_snapshot_data
from load_data_ramses import load_snapshot_ramses

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# dataDir = '/home/bruno/Desktop/data/'
dataDir = '/raid/bruno/data/'

eta = 0.000
beta = 0.10

nPoints = 128
Lbox = 50.0   #Mpc/h

data_name = 'SIMPLE_PLMC_eta0.000_beta0.25_cfl01'

ramsesDir = dataDir + 'cosmo_sims/ramses/{0}_hydro_50Mpc_slope1_cfl2/h5_files/'.format(nPoints )
chollaDir = dataDir + 'cosmo_sims/cholla_pm/{0}_hydro_50Mpc/data_{1}/'.format( nPoints, data_name )
outDir = dev_dir + 'figures/power_hydro/'

# fileName = outDir + 'ps_{0}_hydro_ramses_{2}_eta{1:.3f}.png'.format(  nPoints, eta, reconst )
fileName = outDir + 'ps_{0}_hydro_ramses_{1}.png'.format( nPoints, data_name )

# set simulation volume dimentions
nz, ny, nx = nPoints, nPoints, nPoints
nCells  = nx*ny*nz
h = 0.6766
Lx = Lbox
Ly = Lbox
Lz = Lbox
dx, dy, dz = Lx/(nx), Ly/(ny), Lz/(nz )

n_kSamples = 12



# snapshots = [ 0, 2, 4, 7, 10, 13, 16, 20, 25, 30]

snapshots = [ 0, 2, 4, 7, 10, 13, 16, 20, 24, 28]
# snapshots = [ 0, 2, 4, 7, 10, 13, 16, 19]
# snapshots = [ 0, 2, 4]
n_snapshots = len( snapshots  )

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
fig.set_size_inches(20,10)
fig.clf()


gs = plt.GridSpec(5, 2)
gs.update(hspace=0.05, wspace=0.08, )
ax1 = plt.subplot(gs[0:4, 0])
ax2 = plt.subplot(gs[4:5, 0])

ax3 = plt.subplot(gs[0:4, 1])
ax4 = plt.subplot(gs[4:5, 1])


# colors = ['b', 'y', 'g', 'c', 'm', 'b', 'y', 'g', 'c', 'm', ]
colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

for i in range(n_snapshots):

  ps_dm_cholla = data_all[i,0]
  ps_gas_cholla = data_all[i,1]

  ps_dm_ramses = data_all[i,2]
  ps_gas_ramses = data_all[i,3]



  label = 'z = {0:.1f}'.format(current_z_all[i])
  c = colors[i]
  ax1.plot( k_vals, ps_dm_cholla,  c=c, linewidth=2, label=label )
  ax3.plot( k_vals, ps_gas_cholla, c=c,  linewidth=2, label=label )

  if i == n_snapshots-1:
    ax1.plot( k_vals, ps_dm_ramses, '--', c=c, linewidth=1, label='Ramses' )
    ax3.plot( k_vals, ps_gas_ramses, '--', c=c, linewidth=1, label='Ramses' )
  else:
    ax1.plot( k_vals, ps_dm_ramses, '--', c=c, linewidth=1 )
    ax3.plot( k_vals, ps_gas_ramses, '--', c=c, linewidth=1 )
  #

  error_dm = (ps_dm_cholla - ps_dm_ramses) / ps_dm_cholla
  error_gas = (ps_gas_cholla - ps_gas_ramses) / ps_gas_cholla

  ax2.plot( k_vals, error_dm , c=c, alpha=0.9)
  ax4.plot( k_vals, error_gas , c=c, alpha=0.9)

ax2.axhline( y=0., color='r', linestyle='--',  )
ax2.set_ylim( -0.4, 0.4)
ax4.axhline( y=0., color='r', linestyle='--',  )
ax4.set_ylim( -0.4, 0.4)

ax1.set_ylabel( r'$P(k) $', fontsize=17)
ax2.set_ylabel( 'Difference', fontsize=15)
ax1.legend( loc=3)
ax2.set_xlabel( r'$k \, \, [h Mpc^{-1}]$', fontsize=17)
ax3.legend( loc=3)
ax4.set_xlabel( r'$k \, \, [h Mpc^{-1}]$', fontsize=17)

ax1.set_xscale('log')
ax1.set_yscale('log')
ax3.set_xscale('log')
ax3.set_yscale('log')
ax2.set_xscale('log')
ax4.set_xscale('log')
ax1.set_title('DM Power Spectrum',  fontsize=18)
ax3.set_title('Gas Power Spectrum ',  fontsize=18)


fig.suptitle(r'{0} '.format(data_name ), fontsize=20, y=0.95)

# ax1.xlim()
fig.savefig( fileName,  pad_inches=0.1,  bbox_inches='tight', dpi=80)
print 'Saved Image: ', fileName
