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
from load_data_enzo import load_snapshot_enzo

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

dataDir = '/raid/bruno/data/'


eta_0 = 0.005
beta_0 = 0.01
beta_1 = 0.00

n_arg = len(sys.argv)
if n_arg > 1:
  args = []
  for i in range(1 , n_arg):
    arg = sys.argv[i]
    args.append( float( arg ))
  eta_0, beta_0, beta_1 = args
  if rank == 0:
    print "Using command arguments"
    print args

if rank == 0: print 'eta: {0:.3f}   beta{1:.3f}  {2:.3f}/'.format( eta_0, beta_0, beta_1 )

nPoints = 128
Lbox = 50.0   #Mpc/h

# dataSet = 'PLMP'
enzoDir = dataDir + 'cosmo_sims/enzo/{0}_hydro/h5_files/'.format(nPoints)
chollaDir = dataDir + 'cosmo_sims/cholla_pm/{3}_hydro/data_PPMC_HLLC_SIMPLE_eta{0:.3f}_beta{1:.3f}_{2:.3f}/'.format( eta_0, beta_0, beta_1, nPoints )
outDir = dev_dir + 'figures/power_hydro/'

fileName = outDir + 'ps_{0}_hydro_PPMC_HLLC_SIMPLE_eta{0:.3f}_beta{1:.3f}_{2:.3f}.png'.format( eta_0, beta_0, beta_1, nPoints )

# set simulation volume dimentions
nz, ny, nx = nPoints, nPoints, nPoints
nCells  = nx*ny*nz
h = 0.6774
Lx = Lbox
Ly = Lbox
Lz = Lbox
dx, dy, dz = Lx/(nx), Ly/(ny), Lz/(nz )

n_kSamples = 20



# snapshots = [ 0, 2, 4, 7, 10, 13, 16, 20, 25, 30]

snapshots = [ 0, 2, 4, 7, 10, 13, 16, 20, 24, 27]
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
data_cholla = load_snapshot_data( snapKey, chollaDir, cool=False )
current_z_ch = data_cholla['current_z']
dens_dm_cholla = data_cholla['dm']['density'][...]
dens_gas_cholla = data_cholla['gas']['density'][...]

ps_dm_cholla, k_vals, count_dm_cholla = get_power_spectrum( dens_dm_cholla, Lbox, nx, ny, nz, dx, dy, dz,  n_kSamples=n_kSamples)
ps_gas_cholla, k_vals, count_gas_cholla = get_power_spectrum( dens_gas_cholla, Lbox, nx, ny, nz, dx, dy, dz,  n_kSamples=n_kSamples)

ps_all[0] = ps_dm_cholla
ps_all[1] = ps_gas_cholla


print ' Enzo: ', nSnap
data_enzo = load_snapshot_enzo( nSnap, enzoDir, dm=True, cool=False)
current_a_enzo = data_enzo['current_a']
current_z_enzo = data_enzo['current_z']
dens_dm_enzo = data_enzo['dm']['density'][...]
dens_gas_enzo = data_enzo['gas']['density'][...]

ps_dm_enzo, k_vals, count_dm_enzo = get_power_spectrum( dens_dm_enzo, Lbox, nx, ny, nz, dx, dy, dz,  n_kSamples=n_kSamples)
ps_gas_enzo, k_vals, count_gas_enzo = get_power_spectrum( dens_gas_enzo, Lbox, nx, ny, nz, dx, dy, dz,  n_kSamples=n_kSamples)



ps_all[2] = ps_dm_enzo
ps_all[3] = ps_gas_enzo


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

  ps_dm_enzo = data_all[i,2]
  ps_gas_enzo = data_all[i,3]



  label = 'z = {0:.1f}'.format(current_z_all[i])
  c = colors[i]
  ax1.plot( k_vals, ps_dm_cholla,  c=c, linewidth=2, label=label )
  ax3.plot( k_vals, ps_gas_cholla, c=c,  linewidth=2, label=label )
  
  if i == n_snapshots-1:
    ax1.plot( k_vals, ps_dm_enzo, '--', c=c, linewidth=1, label='Enzo' )
    ax3.plot( k_vals, ps_gas_enzo, '--', c=c, linewidth=1, label='Enzo' )
  else:
    ax1.plot( k_vals, ps_dm_enzo, '--', c=c, linewidth=1 )
    ax3.plot( k_vals, ps_gas_enzo, '--', c=c, linewidth=1 )
  #

  error_dm = (ps_dm_cholla - ps_dm_enzo) / ps_dm_cholla
  error_gas = (ps_gas_cholla - ps_gas_enzo) / ps_gas_cholla
  
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

ax1.set_xscale('log')
ax1.set_yscale('log')
ax3.set_xscale('log')
ax3.set_yscale('log')
ax2.set_xscale('log')
ax1.set_title('DM Power Spectrum',  fontsize=18)
ax3.set_title('Gas Power Spectrum ',  fontsize=18)


fig.suptitle(r'$\eta_0={0:0.3f}$   $\beta_0={1:0.3f}$   $\beta_1={2:0.3f}$'.format( eta_0, beta_0, beta_1 ), fontsize=20, y=0.95)

# ax1.xlim()
fig.savefig( fileName,  pad_inches=0.1,  bbox_inches='tight', dpi=80)
print 'Saved Image: ', fileName
