import sys, os
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
from mpl_toolkits.axes_grid1 import make_axes_locatable
import SCM5




dev_dir = '/home/bruno/Desktop/Dropbox/Developer/'
cosmo_dir = dev_dir + 'cosmo_sims/'
toolsDirectory = cosmo_dir + "tools/"
analysisDirectory = cosmo_dir + "analysis/"
sys.path.extend([toolsDirectory, analysisDirectory ] )
from tools import create_directory
from load_data_cholla import load_snapshot_data_particles
from load_data_nyx import load_snapshot_nyx
from internal_energy import get_temp

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nSnap = rank

dataDir = '/home/bruno/Desktop/hard_drive_1/data/'
# dataDir = '/raid/bruno/data/'



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


nPoints = 256
Lbox = 100000.


chollaDir = dataDir + 'cosmo_sims/cholla_pm/256_dm_50Mpc/'.format(nPoints)
chollaDir_uv = chollaDir +  'data/'

nyxDir = dataDir + 'cosmo_sims/nyx/256_dm_50Mpc/'

outDir = dev_dir + 'figures/dm_projection_50Mpc_Good/'
if rank == 0:
  create_directory( outDir )

proj_offset = 0
proj_depth = 256

def get_projection( data, offset, depth, log=True ):
  if log: proj = np.log10(data[offset:offset+depth, :, :].sum(axis=0))
  else: proj = data[offset:offset+depth, :, :].sum(axis=0)
  return proj



n_snapshots = 323
snapshots = range(0, n_snapshots)

max_all = []
min_all = []
# 
# for nSnap in snapshots:
#   print nSnap
#   # 
#   data_cholla = load_snapshot_data_particles( nSnap, chollaDir_uv )
#   current_z_ch = data_cholla['current_z']
#   current_a_ch = data_cholla['current_a']
#   dens_ch = data_cholla['density'][...]
#   proj_ch = get_projection( dens_ch, proj_offset, proj_depth, log=True )
# 
#   data_enzo = load_snapshot_nyx( nSnap, nyxDir, hydro=False, particles=False )
#   current_a_enzo = data_enzo['dm']['current_a']
#   current_z_enzo = data_enzo['dm']['current_z']
#   dens_en = data_enzo['dm']['density']
#   proj_en = get_projection( dens_en, proj_offset, proj_depth, log=True )
# 
# 
#   max_v = max( proj_en.max(), proj_ch.max() )
#   min_v = min( proj_en.min(), proj_ch.min() )
#   max_all.append( max_v )
#   min_all.append( min_v )
# 
# max_val = max( max_all )
# min_val = min( min_all )
# print min_val, max_val

#64
# min_val, max_val = 2.4101318878571107, 5.752941181680996

#128
min_val, max_val = 2.8934905629282452, 5.758749867867053


nSnap = 258
# for nSnap in range(323):
data_cholla = load_snapshot_data_particles( nSnap, chollaDir_uv )
current_z_ch = data_cholla['current_z']
current_a_ch = data_cholla['current_a']
dens_ch = data_cholla['density'][...]
proj_ch = get_projection( dens_ch, proj_offset, proj_depth, log=True )

data_enzo = load_snapshot_nyx( nSnap, nyxDir, hydro=False, particles=False )
current_a_enzo = data_enzo['dm']['current_a']
current_z_enzo = data_enzo['dm']['current_z']
dens_en = data_enzo['dm']['density']
proj_en = get_projection( dens_en, proj_offset, proj_depth, log=True )




n_rows = 1
n_cols = 2
fig, ax_list = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(10*n_cols,10.5*n_rows))
# fig.clf()
# titles = [ 'Z={0:.2f}   Gas'.format(current_z_ch),  'HI', 'HII', 'Temperature' ]
  # y_labels = [' NYX', 'CHOLLA', 'DIFFERENCE' ]
# 
# 
fs = 18

cmap='bone'
# cmap = 'inferno'
cmap='magma'

ax = ax_list[0]
im = ax.imshow( proj_en, interpolation='bilinear',  vmin=min_val, vmax=max_val, cmap=cmap )
ax.set_title( " NYX".format(current_z_ch), fontsize = fs)
ax.tick_params(axis='both', which='both', bottom=False, labelbottom=False, left=False, labelleft=False)

# place a text box in upper left in axes coords
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
textstr = "z={0:.2f}".format( np.abs(current_z_ch)  )
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=25,
        verticalalignment='top', bbox=props)


ax = ax_list[1]
im = ax.imshow( proj_ch, interpolation='bilinear',  vmin=min_val, vmax=max_val, cmap=cmap )
ax.set_title( "CHOLLA", fontsize = fs)
ax.tick_params(axis='both', which='both', bottom=False, labelbottom=False, left=False, labelleft=False)



fig.tight_layout()
fileName = 'projection_{0}_{1}.png'.format(nSnap, cmap)
fig.savefig( outDir + fileName,  pad_inches=0.1,  bbox_inches='tight', dpi=100 )
print 'Saved image: ', fileName
print ''
