import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import h5py as h5
from mpl_toolkits.axes_grid1 import make_axes_locatable

dataDir = '/raid/bruno/data/'
# dataDir = '/home/bruno/Desktop/hard_drive_1/data/'
dev_dir = '/home/bruno/Desktop/Dropbox/Developer/'
cosmo_dir = dev_dir + 'cosmo_sims/'
toolsDirectory = cosmo_dir + "tools/"
sys.path.extend([toolsDirectory ] )
from tools import *
from load_data_cholla import load_snapshot_data_particles
from load_data_nyx import load_snapshot_nyx
from load_data_enzo import load_snapshot_enzo


outputsDir = '/home/bruno/cholla/scale_output_files/'
nyxDir = dataDir + 'cosmo_sims/nyx/256_dm_50Mpc/'
enzoDir = dataDir + 'cosmo_sims/enzo/256_dm_50Mpc/h5_files/'
chollaDir_enzo = dataDir + 'cosmo_sims/cholla_pm/256_dm_50Mpc/data_enzo/'
chollaDir_nyx = dataDir + 'cosmo_sims/cholla_pm/256_dm_50Mpc/data_nyx/'
outDir = dev_dir + 'figures/comparison_enzo_nyx/'

create_directory( outDir )


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


nSnap = 0

fileName = 'density_dm.png'

proj_offset = 0
proj_depth = 128

def get_projection( data, offset, depth, log=True ):
  if log: proj = np.log10(data[offset:offset+depth, :, :].sum(axis=0))
  else: proj = data[offset:offset+depth, :, :].sum(axis=0)
  return proj




nSnap_enzo = 158

data_cholla = load_snapshot_data_particles( nSnap_enzo, chollaDir_enzo )
current_z_ch_enzo = data_cholla['current_z']
dens_dm_cholla_enzo = data_cholla['density'][...]
proj_cholla_enzo = get_projection( dens_dm_cholla_enzo, proj_offset, proj_depth, log=False )
proj_cholla_enzo_log = np.log10(proj_cholla_enzo)
print " Cholla: ", current_z_ch_enzo, dens_dm_cholla_enzo.mean()

#Load Enzo data
data_enzo = load_snapshot_enzo( nSnap_enzo, enzoDir, dm=True, particles=False, cool=False, metals=False, hydro=False)
current_a_enzo = data_enzo['current_a']
current_z_enzo = data_enzo['current_z']
dens_dm_enzo = data_enzo['dm']['density'][...]
proj_enzo = get_projection( dens_dm_enzo, proj_offset, proj_depth, log=False )
proj_enzo_log = np.log10(proj_enzo)
print ' Enzo: ', current_z_enzo, dens_dm_enzo.mean()

diff_enzo = ( proj_cholla_enzo - proj_enzo ) / proj_enzo

nSnap_nyx = 113

#Load Nyx data
data_nyx = load_snapshot_nyx( nSnap_nyx, nyxDir, hydro=False, particles=False )
current_a_nyx = data_nyx['dm']['current_a']
current_z_nyx = data_nyx['dm']['current_z']
dens_dm_nyx = data_nyx['dm']['density'][...]
proj_nyx = get_projection( dens_dm_nyx, proj_offset, proj_depth, log=False )
proj_nyx_log = np.log10(proj_nyx)
print ' Nyx: ', current_z_nyx, dens_dm_nyx.mean()

data_cholla = load_snapshot_data_particles( nSnap_nyx, chollaDir_nyx )
current_z_ch_nyx = data_cholla['current_z']
dens_dm_cholla_nyx = data_cholla['density'][...]
proj_cholla_nyx = get_projection( dens_dm_cholla_nyx, proj_offset, proj_depth, log=False )
proj_cholla_nyx_log = np.log10(proj_cholla_nyx)
print " Cholla: ", current_z_ch_enzo, dens_dm_cholla_nyx.mean()

diff_nyx = ( proj_cholla_nyx - proj_nyx ) / proj_nyx


print "Enzo:", diff_enzo.min(), diff_enzo.max()
print "Nyx: ", diff_nyx.min(), diff_nyx.max()

n_rows = 2
n_cols = 3
fig, ax_list = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(10*n_cols,9*n_rows))

fs = 18
cmap='magma'


diff_max_enzo = 0.5
diff_max_nyx = 0.005


max_val = max( proj_cholla_nyx_log.max(), proj_nyx_log.max() )
min_val = min( proj_cholla_nyx_log.min(), proj_nyx_log.min() )

ax = ax_list[1][0]
im = ax.imshow( proj_cholla_nyx_log, interpolation='bilinear',  vmin=min_val, vmax=max_val, cmap=cmap )
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar( im, cax=cax )
ax.set_title( "CHOLLA", fontsize = fs)
ax.tick_params(axis='both', which='both', bottom=False, labelbottom=False, left=False, labelleft=False)

ax = ax_list[1][1]
im = ax.imshow( proj_nyx_log, interpolation='bilinear',  vmin=min_val, vmax=max_val, cmap=cmap )
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar( im, cax=cax )
ax.set_title( "NYX", fontsize = fs)
ax.tick_params(axis='both', which='both', bottom=False, labelbottom=False, left=False, labelleft=False)



ax = ax_list[1][2]
im = ax.imshow( diff_nyx, interpolation='bilinear',  vmin=-1*diff_max_nyx, vmax=1*diff_max_nyx, cmap='bwr' )
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar( im, cax=cax )
ax.set_title( "DIFFERENCE: CHOLLA-NYX", fontsize = fs)
ax.tick_params(axis='both', which='both', bottom=False, labelbottom=False, left=False, labelleft=False)


max_val = max( proj_cholla_enzo_log.max(), proj_enzo_log.max() )
min_val = min( proj_cholla_enzo_log.min(), proj_enzo_log.min() )


ax = ax_list[0][0]
im = ax.imshow( proj_cholla_enzo_log, interpolation='bilinear',  vmin=min_val, vmax=max_val, cmap=cmap )
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar( im, cax=cax )
ax.set_title( "CHOLLA", fontsize = fs)
ax.tick_params(axis='both', which='both', bottom=False, labelbottom=False, left=False, labelleft=False)

ax = ax_list[0][1]
im = ax.imshow( proj_enzo_log, interpolation='bilinear',  vmin=min_val, vmax=max_val, cmap=cmap )
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar( im, cax=cax )
ax.set_title( "ENZO", fontsize = fs)
ax.tick_params(axis='both', which='both', bottom=False, labelbottom=False, left=False, labelleft=False)

ax = ax_list[0][2]
im = ax.imshow( diff_enzo, interpolation='bilinear',  vmin=-1*diff_max_enzo, vmax=1*diff_max_enzo, cmap='bwr' )
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar( im, cax=cax )
ax.set_title( "DIFFERENCE: CHOLLA-ENZO", fontsize = fs)
ax.tick_params(axis='both', which='both', bottom=False, labelbottom=False, left=False, labelleft=False)

v_max = max( proj_cholla_enzo_log.max(), proj_enzo_log.max() )
v_min = min( proj_cholla_enzo_log.min(), proj_enzo_log.min() )




fig.suptitle('z=0  DM Density Comparison', fontsize=22, y=0.93)
fig.savefig( outDir + fileName,   bbox_inches='tight', dpi=100)
print 'Saved image: ', fileName
print ''