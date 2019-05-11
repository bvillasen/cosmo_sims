import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
from mpl_toolkits.axes_grid1 import make_axes_locatable

cosmo_dir = '/home/bruno/Desktop/Dropbox/Developer/cosmo_sims/'
outDir = cosmo_dir + 'figures/power_hydro/'
toolsDirectory = cosmo_dir + "tools/"
sys.path.extend([toolsDirectory ] )
from load_data_cholla import load_snapshot_data
from load_data_enzo import load_snapshot_enzo
from tools import create_directory

rank = 0

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nSnap = rank


name = 'de1_PPMC_HLLC'

outDir = cosmo_dir + 'figures/projections/{0}/'.format(name)

dataDir = '/raid/bruno/data/'

n_rows = 3
chollaDir_0 = dataDir + 'cosmo_sims/cholla_pm/256_cool/data_{0}/'.format(name)
chollaDir_1 = dataDir + 'cosmo_sims/cholla_pm/256_cool/data_de001_PCM_noGrav_noPFlux/'
chollaDir_2 = dataDir + 'cosmo_sims/cholla_pm/256_cool/data_de001_PCM_noCool/'
chollaDir_3 = dataDir + 'cosmo_sims/cholla_pm/256_cool/data_de02_PPMC_noFirst/'
# chollaDir_4 = dataDir + 'cosmo_sims/cholla_pm/256_cool/data_de02_PPMC/'


if rank == 0: 
  create_directory( outDir )
  print "Output: ", outDir


comm.Barrier()

enzoDir_uv = dataDir + 'cosmo_sims/enzo/256_cool_uv/h5_files/'

metals = True

start_indx = 128
n_slices = 5

def get_prjection( field ):
  proj = field[start_indx:start_indx+n_slices, :, :].sum(axis=0)
  return proj


gamma = 5./3
nPoints = 256
nx = nPoints
ny = nPoints
nz = nPoints

dv = (115000./256)**3

slice_0 = 0
n_slice = 64

dens_weight = True

t_min, d_min = 1e20, 1e20
t_max, d_max = -1e20, -1e20

nSnap = rank

n_cols = 8


data_cholla = load_snapshot_data( nSnap, chollaDir_0, cool=True )
current_z_ch = data_cholla['current_z']
current_a_ch = data_cholla['current_a']
dens_ch = data_cholla['gas']['density'][...]
vel_x_ch = data_cholla['gas']['momentum_x'][...]
vel_y_ch = data_cholla['gas']['momentum_y'][...]
vel_z_ch = data_cholla['gas']['momentum_z'][...]
E_ch = data_cholla['gas']['Energy'][...]
Ekin_ch = 0.5 * ( vel_x_ch*vel_x_ch + vel_y_ch*vel_y_ch + vel_z_ch*vel_z_ch  ) / dens_ch
Utot_ch =  E_ch - Ekin_ch
U_ch = data_cholla['gas']['GasEnergy'][...]
d_proj_ch = get_prjection( dens_ch )
vx_proj_ch = get_prjection( vel_x_ch )
vy_proj_ch = get_prjection( vel_y_ch )
vz_proj_ch = get_prjection( vel_z_ch )
E_proj_ch = get_prjection( E_ch )
U_proj_ch = get_prjection( U_ch )
Utot_proj_ch = get_prjection( Utot_ch )
Ekin_proj_ch = get_prjection( Ekin_ch )

data_ch = [ np.log10(d_proj_ch), np.log10(np.abs(vx_proj_ch)), np.log10(np.abs(vy_proj_ch)), np.log10(np.abs(vz_proj_ch)), np.log10(E_proj_ch), np.log10(Ekin_proj_ch), np.log10(Utot_proj_ch), np.log10(U_proj_ch) ]

data_enzo = load_snapshot_enzo( nSnap, enzoDir_uv, cool=True, metals=metals )
current_a_enzo = data_enzo['current_a']
current_z_enzo = data_enzo['current_z']
dens_en = data_enzo['gas']['density'][...]
vel_x_en = data_enzo['gas']['momentum_x'][...]
vel_y_en = data_enzo['gas']['momentum_y'][...]
vel_z_en = data_enzo['gas']['momentum_z'][...]
E_en = data_enzo['gas']['Energy'][...]
Ekin_en = 0.5 * ( vel_x_en*vel_x_en + vel_y_en*vel_y_en + vel_z_en*vel_z_en  ) / dens_en
Utot_en =  E_en - Ekin_en
U_en = data_enzo['gas']['GasEnergy'][...]
d_proj_en = get_prjection( dens_en )
vx_proj_en = get_prjection( vel_x_en )
vy_proj_en = get_prjection( vel_y_en )
vz_proj_en = get_prjection( vel_z_en )
E_proj_en = get_prjection( E_en )
U_proj_en = get_prjection( U_en )
Utot_proj_en = get_prjection( Utot_en )
Ekin_proj_en = get_prjection( Ekin_en )

data_en = [ np.log10(d_proj_en), np.log10(np.abs(vx_proj_en)), np.log10(np.abs(vy_proj_en)), np.log10(np.abs(vz_proj_en)), np.log10(E_proj_en), np.log10(Ekin_proj_en), np.log10(Utot_proj_en), np.log10(U_proj_en) ]

d_diff = (d_proj_ch - d_proj_en ) / d_proj_en
vx_diff = (vx_proj_ch - vx_proj_en ) / vx_proj_en
vy_diff = (vy_proj_ch - vy_proj_en ) / vy_proj_en
vz_diff = (vz_proj_ch - vz_proj_en ) / vz_proj_en
E_diff = (E_proj_ch - E_proj_en ) / E_proj_en
U_diff = (U_proj_ch - U_proj_en ) / U_proj_en
Utot_diff = (Utot_proj_ch - Utot_proj_en ) / Utot_proj_en
Ekin_diff = (Ekin_proj_ch - Ekin_proj_en ) / Ekin_proj_en

data_diff = [ d_diff, vx_diff, vy_diff, vz_diff, E_diff, Ekin_diff, Utot_diff, U_diff ]

data_max = [max( data_en[i].max(), data_ch[i].max() ) for i in range(n_cols) ]
data_min = [min( data_en[i].min(), data_ch[i].min() ) for i in range(n_cols) ]



types = ['enzo', 'cholla', 'diff']
data = {}
data['enzo'] = data_en
data['cholla'] = data_ch
data['diff'] = data_diff

fig, ax_list = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(10*n_cols,10*n_rows))
titles = [ 'Z={0:.2f}  log( Density )'.format(current_z_ch), 'log( Momentum X )', 'log( Momentum Y )', 'log( Momentum Z )', 'log( Energy )',  'log( Ekin )' , 'log( E - Ekin )',  'log( GasEnergy )' ]
y_labels = [' ENZO ', 'CHOLLA ', 'Difference']

for row in range(n_rows):
  type = types[row]
  for i in range( n_cols):
    ax = ax_list[row][i]
    vmax = data_max[i]
    vmin = data_min[i]
    if row == 2:
      vmax = 0.5
      vmin = -0.5
    if row == 2 :im = ax.imshow( data[type][i], interpolation='bilinear', vmin=vmax, vmax=vmin, cmap='coolwarm' )
    else :im = ax.imshow( data[type][i], interpolation='bilinear', vmin=vmax, vmax=vmin, cmap='jet' )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar( im, cax=cax )
    ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    if i==0 : ax.set_ylabel( y_labels[row], fontsize=30)
    if row == 0: ax.set_title( titles[i], fontsize=35)


fig.tight_layout()
fileName = 'projections_{0}.png'.format(nSnap)
fig.savefig( outDir + fileName )
print 'Saved image: ', fileName
print ''
