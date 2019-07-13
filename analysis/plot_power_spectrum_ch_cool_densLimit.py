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
#
# from mpi4py import MPI
#
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()

rank = 0

dataDir = '/raid/bruno/data/'


eta_0 = 0.005
beta_0 = 0.12
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

extra_name = ''

chollaDir = dataDir + 'cosmo_sims/cholla_pm/{3}_cool/data_PPMC_HLLC_SIMPLE_eta{0:.3f}_beta{1:.3f}_{2:.3f}{4}/'.format( eta_0, beta_0, beta_1, nPoints, extra_name )

enzoDir = dataDir + 'cosmo_sims/enzo/{0}_cool_uv/h5_files/'.format(nPoints)
outDir = dev_dir + 'figures/power_hydro/'

fileName = outDir + 'ps_{0}_cooling_uv_PPMC_HLLC_SIMPLE_eta{0:.3f}_beta{1:.3f}_{2:.3f}{4}_densLimit.png'.format( eta_0, beta_0, beta_1, nPoints, extra_name )

# set simulation volume dimentions
nz, ny, nx = nPoints, nPoints, nPoints
nCells  = nx*ny*nz
h = 0.6774
Lx = Lbox
Ly = Lbox
Lz = Lbox
dx, dy, dz = Lx/(nx), Ly/(ny), Lz/(nz )

n_kSamples = 20

def apply_upper_limit( data_input, limit_val, mean_val=None ):
  data = data_input.copy()
  if mean_val == None: mean_val = data.mean()
  indxs = np.where( data > limit_val* mean_val )
  data[indxs] = limit_val * mean_val
  return data


nSnap = 27


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



print ' Enzo: ', nSnap
data_enzo = load_snapshot_enzo( nSnap, enzoDir, dm=True, cool=True)
current_a_enzo = data_enzo['current_a']
current_z_enzo = data_enzo['current_z']
dens_dm_enzo = data_enzo['dm']['density'][...]
dens_gas_enzo = data_enzo['gas']['density'][...]
dens_gas_H_enzo = data_enzo['gas']['HI_density'][...]
dens_gas_HII_enzo = data_enzo['gas']['HII_density'][...]
# dens_gas_H_enzo *= dens_gas_enzo.mean() / dens_gas_H_enzo.mean()


fig = plt.figure(0)
fig.set_size_inches(40,10)
fig.clf()

fig.suptitle(r'z={4:.2f}    $\eta_0={0:0.3f}$   $\beta_0={1:0.3f}$   $\beta_1={2:0.3f}$  {3}'.format( eta_0, beta_0, beta_1, extra_name, np.abs(current_z_ch) ), fontsize=20, y=0.95)

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

colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']




upper_limit_vals = [ 2, 10, 100, 1000, 10000 ]
n_vals = len( upper_limit_vals )
for i,upper_limit in enumerate(upper_limit_vals):
  print upper_limit

  dens_dm_ch = apply_upper_limit( dens_dm_cholla, upper_limit )
  dens_gas_ch = apply_upper_limit( dens_gas_cholla, upper_limit )
  dens_gas_H_ch = apply_upper_limit( dens_gas_H_cholla, upper_limit )
  dens_gas_HII_ch = apply_upper_limit( dens_gas_HII_cholla, upper_limit )

  dens_dm_en = apply_upper_limit( dens_dm_enzo, upper_limit )
  dens_gas_en = apply_upper_limit( dens_gas_enzo, upper_limit )
  dens_gas_H_en = apply_upper_limit( dens_gas_H_enzo, upper_limit )
  dens_gas_HII_en = apply_upper_limit( dens_gas_HII_enzo, upper_limit )


  ps_dm_cholla, k_vals, count_dm_cholla = get_power_spectrum( dens_dm_ch, Lbox, nx, ny, nz, dx, dy, dz,  n_kSamples=n_kSamples)
  ps_gas_cholla, k_vals, count_gas_cholla = get_power_spectrum( dens_gas_ch, Lbox, nx, ny, nz, dx, dy, dz,  n_kSamples=n_kSamples)
  ps_gas_H_cholla, k_vals, count_gas_cholla = get_power_spectrum( dens_gas_H_ch, Lbox, nx, ny, nz, dx, dy, dz,  n_kSamples=n_kSamples)
  ps_gas_HII_cholla, k_vals, count_gas_cholla = get_power_spectrum( dens_gas_HII_ch, Lbox, nx, ny, nz, dx, dy, dz,  n_kSamples=n_kSamples)

  ps_dm_enzo, k_vals, count_dm_enzo = get_power_spectrum( dens_dm_en, Lbox, nx, ny, nz, dx, dy, dz,  n_kSamples=n_kSamples)
  ps_gas_enzo, k_vals, count_gas_enzo = get_power_spectrum( dens_gas_en, Lbox, nx, ny, nz, dx, dy, dz,  n_kSamples=n_kSamples)
  ps_gas_H_enzo, k_vals, count_gas_enzo = get_power_spectrum( dens_gas_H_en, Lbox, nx, ny, nz, dx, dy, dz,  n_kSamples=n_kSamples)
  ps_gas_HII_enzo, k_vals, count_gas_enzo = get_power_spectrum( dens_gas_HII_en, Lbox, nx, ny, nz, dx, dy, dz,  n_kSamples=n_kSamples)

  label = r'$\alpha = {0:.1f} $'.format(upper_limit)
  c = colors[i]
  ax1.plot( k_vals, ps_dm_cholla,  c=c, linewidth=2, label=label )
  ax3.plot( k_vals, ps_gas_cholla, c=c,  linewidth=2, label=label )
  ax5.plot( k_vals, ps_gas_H_cholla, c=c,  linewidth=2, label=label )
  ax7.plot( k_vals, ps_gas_HII_cholla, c=c,  linewidth=2, label=label )

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
#
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
ax1.legend( loc=3, fontsize=15)
ax2.set_xlabel( r'$k \, \, [h Mpc^{-1}]$', fontsize=17)
ax3.legend( loc=3, fontsize=15)
ax4.set_xlabel( r'$k \, \, [h Mpc^{-1}]$', fontsize=17)
ax5.legend( loc=3, fontsize=15)
ax6.set_xlabel( r'$k \, \, [h Mpc^{-1}]$', fontsize=17)
ax7.legend( loc=3, fontsize=15)
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
ax1.set_title('DM Power Spectrum',  fontsize=18)
ax3.set_title('Gas Power Spectrum ',  fontsize=18)
ax5.set_title('Neutral Hydrogen Power Spectrum',  fontsize=18)
ax7.set_title('Ionized Hydrogen Power Spectrum',  fontsize=18)
#
#
#
# ax1.xlim()
fig.savefig( fileName,  pad_inches=0.1,  bbox_inches='tight', dpi=80)
print 'Saved Image: ', fileName
