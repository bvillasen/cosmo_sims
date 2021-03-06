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


eta_1 = 0.001
eta_2 = 0.0500


n_arg = len(sys.argv)
if n_arg > 1:
  args = []
  for i in range(1 , n_arg):
    arg = sys.argv[i]
    args.append( float( arg ))
  eta_1, eta_2, n_out = args
  n_out = int( n_out)
  if rank == 0:
    print "Using command arguments"
    print args

if rank == 0: print 'eta: {0:.3f}  {1:.3f}  '.format( eta_1, eta_2, )

nPoints = 128
Lbox = 50.0   #Mpc/h

integrator = 'SIMPLE'
extra_name = ''

chollaDir = dataDir + 'cosmo_sims/cholla_pm/{3}_cool/data_SIMPLE_PPMP_eta0.035_0.00_grav4_clean/'.format( eta_1, eta_2,  extra_name, nPoints, integrator )

enzoDir = dataDir + 'cosmo_sims/enzo/{0}_cool_uv_50Mpc_HLLC_grav4/h5_files/'.format(nPoints)
outDir = dev_dir + 'figures/power_hydro/'

fileName = outDir + 'ps_{2}_cooling_uv_PPMC_HLLC_new.png'.format( eta_1, eta_2, nPoints, extra_name, integrator  )
# fileName = outDir + 'ps_{2}_cooling_uv_PPMC_HLLC_{4}_{5}.png'.format( eta_1, eta_2, nPoints, extra_name, integrator, n_out  )



# set simulation volume dimentions
nz, ny, nx = nPoints, nPoints, nPoints
nCells  = nx*ny*nz
h = 0.6774
Lx = Lbox
Ly = Lbox
Lz = Lbox
dx, dy, dz = Lx/(nx), Ly/(ny), Lz/(nz )

n_kSamples = 20



# snapshots = [ 0, 2, 4, 7, 10, 13, 16, 20, 25, 31]

snapshots = [ 0, 2, 4, 7, 10, 13, 16, 22, 24, 27]
# snapshots = [ 0, 2, 4]
n_snapshots = len( snapshots  )

if rank >= n_snapshots: exit()
nSnap = snapshots[rank]

n_power_data = 8
ps_all = np.ones( [n_power_data, n_kSamples] ) 
# ps_all *= rank

print " Cholla: ", nSnap
snapKey = str( nSnap )
# if i not in [9]: continue
data_cholla = load_snapshot_data( snapKey, chollaDir, cool=True )
current_z_ch = data_cholla['current_z']
dens_dm_cholla = data_cholla['dm']['density'][...]
dens_gas_cholla = data_cholla['gas']['density'][...]
dens_gas_H_cholla = data_cholla['gas']['HI_density'][...]
dens_gas_HII_cholla = data_cholla['gas']['HII_density'][...]
# dens_gas_H_cholla *= dens_gas_cholla.mean() / dens_gas_H_cholla.mean()

ps_dm_cholla, k_vals, count_dm_cholla = get_power_spectrum( dens_dm_cholla, Lbox, nx, ny, nz, dx, dy, dz,  n_kSamples=n_kSamples)
ps_gas_cholla, k_vals, count_gas_cholla = get_power_spectrum( dens_gas_cholla, Lbox, nx, ny, nz, dx, dy, dz,  n_kSamples=n_kSamples)
ps_gas_H_cholla, k_vals, count_gas_cholla = get_power_spectrum( dens_gas_H_cholla, Lbox, nx, ny, nz, dx, dy, dz,  n_kSamples=n_kSamples)
ps_gas_HII_cholla, k_vals, count_gas_cholla = get_power_spectrum( dens_gas_HII_cholla, Lbox, nx, ny, nz, dx, dy, dz,  n_kSamples=n_kSamples)

ps_all[0] = ps_dm_cholla
ps_all[1] = ps_gas_cholla
ps_all[2] = ps_gas_H_cholla
ps_all[3] = ps_gas_HII_cholla


print ' Enzo: ', nSnap
data_enzo = load_snapshot_enzo( nSnap, enzoDir, dm=True, cool=True)
current_a_enzo = data_enzo['current_a']
current_z_enzo = data_enzo['current_z']
dens_dm_enzo = data_enzo['dm']['density'][...]
dens_gas_enzo = data_enzo['gas']['density'][...]
dens_gas_H_enzo = data_enzo['gas']['HI_density'][...]
dens_gas_HII_enzo = data_enzo['gas']['HII_density'][...]
# dens_gas_H_enzo *= dens_gas_enzo.mean() / dens_gas_H_enzo.mean()

ps_dm_enzo, k_vals, count_dm_enzo = get_power_spectrum( dens_dm_enzo, Lbox, nx, ny, nz, dx, dy, dz,  n_kSamples=n_kSamples)
ps_gas_enzo, k_vals, count_gas_enzo = get_power_spectrum( dens_gas_enzo, Lbox, nx, ny, nz, dx, dy, dz,  n_kSamples=n_kSamples)
ps_gas_H_enzo, k_vals, count_gas_enzo = get_power_spectrum( dens_gas_H_enzo, Lbox, nx, ny, nz, dx, dy, dz,  n_kSamples=n_kSamples)
ps_gas_HII_enzo, k_vals, count_gas_enzo = get_power_spectrum( dens_gas_HII_enzo, Lbox, nx, ny, nz, dx, dy, dz,  n_kSamples=n_kSamples)



ps_all[4] = ps_dm_enzo
ps_all[5] = ps_gas_enzo
ps_all[6] = ps_gas_H_enzo
ps_all[7] = ps_gas_HII_enzo


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

# fig.suptitle(r'$\eta_1={0:0.3f}$   $\eta_2={1:0.4f}$  {3}  {2}'.format( eta_1, eta_2,  extra_name, integrator ), fontsize=20, y=0.95)

gs = plt.GridSpec(5, 2)
gs.update(hspace=0.05, wspace=0.08, )
ax1 = plt.subplot(gs[0:4, 0])
ax2 = plt.subplot(gs[4:5, 0])

ax3 = plt.subplot(gs[0:4, 1])
ax4 = plt.subplot(gs[4:5, 1])

# ax5 = plt.subplot(gs[0:4, 2])
# ax6 = plt.subplot(gs[4:5, 2])
# 
# ax7 = plt.subplot(gs[0:4, 3])
# ax8 = plt.subplot(gs[4:5, 3])


# colors = ['b', 'y', 'g', 'c', 'm', 'b', 'y', 'g', 'c', 'm', ]
colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

for j in range(n_snapshots):
  i = n_snapshots - j -1

  ps_dm_cholla = data_all[i,0]
  ps_gas_cholla = data_all[i,1]
  ps_gas_H_cholla = data_all[i,2]
  ps_gas_HII_cholla = data_all[i,3]

  ps_dm_enzo = data_all[i,4]
  ps_gas_enzo = data_all[i,5]
  ps_gas_H_enzo = data_all[i,6]
  ps_gas_HII_enzo = data_all[i,7]



  label = 'z = {0:.1f}'.format(np.abs(current_z_all[i]))
  c = colors[i]

  if i == n_snapshots-1:
    ax1.plot( k_vals, ps_dm_enzo, '--', c=c, linewidth=1, label='Enzo' )
    ax3.plot( k_vals, ps_gas_enzo, '--', c=c, linewidth=1, label='Enzo' )
    # ax5.plot( k_vals, ps_gas_H_enzo, '--', c=c, linewidth=1, label='Enzo' )
    # ax7.plot( k_vals, ps_gas_HII_enzo, '--', c=c, linewidth=1, label='Enzo' )
  else:
    ax1.plot( k_vals, ps_dm_enzo, '--', c=c, linewidth=1 )
    ax3.plot( k_vals, ps_gas_enzo, '--', c=c, linewidth=1 )
    # ax5.plot( k_vals, ps_gas_H_enzo, '--', c=c, linewidth=1  )
    # ax7.plot( k_vals, ps_gas_HII_enzo, '--', c=c, linewidth=1 )
  #
  ax1.plot( k_vals, ps_dm_cholla,  c=c, linewidth=2, label=label )
  ax3.plot( k_vals, ps_gas_cholla, c=c,  linewidth=2, label=label )
  # ax5.plot( k_vals, ps_gas_H_cholla, c=c,  linewidth=2, label=label )
  # ax7.plot( k_vals, ps_gas_HII_cholla, c=c,  linewidth=2, label=label )

  error_dm = (ps_dm_cholla - ps_dm_enzo) / ps_dm_cholla
  error_gas = (ps_gas_cholla - ps_gas_enzo) / ps_gas_cholla
  error_gas_H = (ps_gas_H_cholla - ps_gas_H_enzo) / ps_gas_H_cholla
  error_gas_HII = (ps_gas_HII_cholla - ps_gas_HII_enzo) / ps_gas_HII_cholla

  ax2.plot( k_vals, error_dm , c=c, alpha=0.9)
  ax4.plot( k_vals, error_gas , c=c, alpha=0.9)
  # ax6.plot( k_vals, error_gas_H , c=c, alpha=0.9)
  # ax8.plot( k_vals, error_gas_HII , c=c, alpha=0.9)

y_min = -1
y_max = 1

ax2.axhline( y=0., color='r', linestyle='--',  )
ax2.set_ylim( y_min, y_max)
ax4.axhline( y=0., color='r', linestyle='--',  )
ax4.set_ylim( y_min, y_max)
# ax6.axhline( y=0., color='r', linestyle='--',  )
# ax6.set_ylim( y_min, y_max)
# ax8.axhline( y=0., color='r', linestyle='--',  )
# ax8.set_ylim( y_min, y_max)

ax1.set_ylabel( r'$P(k) \,\,\, [h^3 Mpc^{-3}] $', fontsize=17)
ax2.set_ylabel( 'Difference', fontsize=15)
ax1.legend( loc=3, fontsize=12)
ax2.set_xlabel( r'$k \, \, [h Mpc^{-1}]$', fontsize=17)
ax3.legend( loc=3, fontsize=12)
ax4.set_xlabel( r'$k \, \, [h Mpc^{-1}]$', fontsize=17)
# ax5.legend( loc=3)
# ax6.set_xlabel( r'$k \, \, [h Mpc^{-1}]$', fontsize=17)
# ax7.legend( loc=3)
# ax8.set_xlabel( r'$k \, \, [h Mpc^{-1}]$', fontsize=17)

ax1.set_xscale('log')
ax1.set_yscale('log')
ax3.set_xscale('log')
ax3.set_yscale('log')
# ax5.set_xscale('log')
# ax5.set_yscale('log')
# ax7.set_xscale('log')
# ax7.set_yscale('log')
ax2.set_xscale('log')
ax4.set_xscale('log')
# ax6.set_xscale('log')
# ax8.set_xscale('log')
ax1.set_title('DM Power Spectrum',  fontsize=18)
ax3.set_title('Gas Power Spectrum ',  fontsize=18)
# ax5.set_title('Neutral Hydrogen Power Spectrum',  fontsize=18)
# ax7.set_title('Ionized Hydrogen Power Spectrum',  fontsize=18)



# ax1.xlim()
fig.savefig( fileName,  pad_inches=0.1,  bbox_inches='tight', dpi=100)
print 'Saved Image: ', fileName
