import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

# dataDir = '/home/bruno/Desktop/data/'
dataDir = '/raid/bruno/data/'



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

outDir = dev_dir + 'figures/dm_projection_50Mpc/'
if rank == 0:
  create_directory( outDir )

proj_offset = 0
proj_depth = 128

def get_projection( data, offset, depth, log=True ):
  if log: proj = np.log10(data[offset:offset+depth, :, :].sum(axis=0))
  else: proj = data[offset:offset+depth, :, :].sum(axis=0)
  return proj


fields = ['density' ]

# nSnap = 0
n_snapshots = 323
snapshots = range(0, n_snapshots)

max_all = 0
min_all = 100

for nSnap in snapshots:
  # 
  data_ch = {}
  data_cholla = load_snapshot_data_particles( nSnap, chollaDir_uv )
  current_z_ch = data_cholla['current_z']
  current_a_ch = data_cholla['current_a']
  for field in fields:
    data = data_cholla[field][...]
    data_ch[field] = {}
    proj = get_projection( data, proj_offset, proj_depth )
    data_ch[field]['proj'] = proj
    data_ch[field]['max'] = proj.max()
    data_ch[field]['min'] = proj.min() 


  data_en = {}
  data_enzo = load_snapshot_nyx( nSnap, nyxDir, hydro=False, particles=False )
  current_a_enzo = data_enzo['dm']['current_a']
  current_z_enzo = data_enzo['dm']['current_z']
  for field in fields:
    data = data_enzo['dm'][field][...]
    data_en[field] = {}
    proj = get_projection( data, proj_offset, proj_depth )
    data_en[field]['proj'] = proj
    data_en[field]['max'] = proj.max()
    data_en[field]['min'] = proj.min() 

  data_diff = {}
  for field in fields:

    vals_ch = data_cholla[field][...]  
    vals_en = data_enzo['dm'][field][...]
    data_diff[field] = {}
    proj_en = get_projection( vals_en, proj_offset, proj_depth, log=False )
    proj_ch = get_projection( vals_ch, proj_offset, proj_depth, log=False )
    proj = ( proj_ch - proj_en ) / proj_en 
    data_diff[field]['proj'] = proj
    data_diff[field]['max'] = proj.max()
    data_diff[field]['min'] = proj.min() 
    max_all = max( max_all, proj.max() )
    min_all = min( min_all, proj.min() )
    print max_all
    print min_all

  
  data_all = [ data_en, data_ch, data_diff ]
  
  n_rows = 1
  n_cols = 3
  fig, ax_list = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(10*n_cols,10*n_rows))
  titles = [ 'Z={0:.2f}   Gas'.format(current_z_ch),  'HI', 'HII', 'Temperature' ]
  y_labels = [' NYX', 'CHOLLA', 'DIFFERENCE' ]
  
  max_val = max( data_en['density']['proj'].max(), data_ch['density']['proj'].max())
  min_val = min( data_en['density']['proj'].min(), data_ch['density']['proj'].min())
  
  
  fs = 18
  
  ax = ax_list[0]
  proj = data_en['density']['proj']
  im = ax.imshow( proj, interpolation='bilinear',  vmin=min_val, vmax=max_val )
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="5%", pad=0.05)
  fig.colorbar( im, cax=cax )
  ax.set_title( "z={0:.2f}   NYX".format(current_z_ch), fontsize = fs)
  
  ax = ax_list[1]
  proj = data_ch['density']['proj']
  im = ax.imshow( proj, interpolation='bilinear',  vmin=min_val, vmax=max_val )
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="5%", pad=0.05)
  fig.colorbar( im, cax=cax )
  ax.set_title( "CHOLLA", fontsize = fs)
  
  ax = ax_list[2]
  proj = data_diff['density']['proj']
  im = ax.imshow( proj, interpolation='bilinear',  vmin=-0.006, vmax=0.006, cmap='jet' )
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="5%", pad=0.05)
  fig.colorbar( im, cax=cax )
  ax.set_title( "Fractional Difference" , fontsize = fs)
  
  
  fig.tight_layout()
  fileName = 'projection_{0}.png'.format(nSnap)
  fig.savefig( outDir + fileName )
  print 'Saved image: ', fileName
  print ''
