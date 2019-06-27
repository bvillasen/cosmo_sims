import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
from mpl_toolkits.axes_grid1 import make_axes_locatable

dev_dir = '/home/bruno/Desktop/Dropbox/Developer/'
cosmo_dir = dev_dir + 'cosmo_sims/'
toolsDirectory = cosmo_dir + "tools/"
sys.path.extend([toolsDirectory ] )
from tools import create_directory
from load_data_cholla import load_snapshot_data
from load_data_enzo import load_snapshot_enzo

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nSnap = rank

# dataDir = '/home/bruno/Desktop/data/'
dataDir = '/raid/bruno/data/'

eta_0 = 0.001
beta_0 = 0.1
beta_1 = 0.02

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


print 'eta: {0:.3f}   beta{1:.3f}  {2:.3f}/'.format( eta_0, beta_0, beta_1 )

chollaDir = dataDir + 'cosmo_sims/cholla_pm/256_cool/'
chollaDir_uv = chollaDir +  'data_PPMC_HLLC_SIMPLE_eta{0:.3f}_beta{1:.3f}_{2:.3f}/'.format( eta_0, beta_0, beta_1 )

enzoDir = dataDir + 'cosmo_sims/enzo/'
enzoDir_uv = enzoDir + '256_cool_uv/h5_files/'


outDir = dev_dir + 'figures/chemistry/chemistry_HI_eta{0:.3f}_beta{1:.3f}_{2:.3f}/'.format( eta_0, beta_0, beta_1 )
if rank == 0:
  create_directory( outDir )


metals = True

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

def get_projection( data, offset, depth, log=True ):
  if log: proj = np.log10(data[offset:offset+depth, :, :].sum(axis=0))
  else: proj = data[offset:offset+depth, :, :].sum(axis=0)
  return proj


proj_offset = 0
proj_depth = 10

fields = ['density', 'HI_density', 'HII_density', 'temperature' ]

# nSnap = 0
# n_snapshots = 10
# snapshots = range(0, n_snapshots)
# for nSnap in snapshots:
# 
data_ch = {}
data_cholla = load_snapshot_data( nSnap, chollaDir_uv, cool=True )
current_z_ch = data_cholla['current_z']
current_a_ch = data_cholla['current_a']
for field in fields:
  data = data_cholla['gas'][field][...]
  data_ch[field] = {}
  proj = get_projection( data, proj_offset, proj_depth )
  data_ch[field]['proj'] = proj
  data_ch[field]['max'] = proj.max()
  data_ch[field]['min'] = proj.min() 


data_en = {}
data_enzo = load_snapshot_enzo( nSnap, enzoDir_uv, cool=True, metals=metals )
current_a_enzo = data_enzo['current_a']
current_z_enzo = data_enzo['current_z']
for field in fields:
  data = data_enzo['gas'][field][...]
  data_en[field] = {}
  proj = get_projection( data, proj_offset, proj_depth )
  data_en[field]['proj'] = proj
  data_en[field]['max'] = proj.max()
  data_en[field]['min'] = proj.min() 

data_diff = {}
for field in fields:
  diff = ( data_cholla['gas'][field][...] - data_enzo['gas'][field][...] ) / data_enzo['gas'][field][...]
  data_diff[field] = {}
  proj = get_projection( diff, proj_offset, proj_depth, log=False )
  data_diff[field]['proj'] = proj
  data_diff[field]['max'] = proj.max()
  data_diff[field]['min'] = proj.min() 



data_all = [ data_en, data_ch, data_diff ]

n_rows = 3
n_cols = len(fields)
fig, ax_list = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(10*n_cols,10*n_rows))
titles = [ 'Z={0:.2f}   Gas'.format(current_z_ch),  'HI', 'HII', 'Temperature' ]
y_labels = [' ENZO', 'CHOLLA', 'DIFFERENCE' ]

for i in range( n_cols):
  field = fields[i]
  min_val = min( data_en[field]['min'], data_ch[field]['min'], )
  max_val = max( data_en[field]['max'], data_ch[field]['max'], )


  for n in range(n_rows):
    data = data_all[n]


    ax = ax_list[n][i]
    proj = data[field]['proj']

    if n == n_rows-1:
      # min_val = max( -10, proj.min())
      # max_val = min( 10 , proj.max() )
      min_val = -10
      max_val = 10
      im = ax.imshow( proj, interpolation='bilinear',  vmin=min_val, vmax=max_val, cmap='jet' )
      
    else:
      if field=='temperature': im = ax.imshow( proj, interpolation='bilinear',  vmin=min_val, vmax=max_val, cmap='jet' )
      else : im = ax.imshow( proj, interpolation='bilinear', vmin=min_val, vmax=max_val )
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar( im, cax=cax )
    ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    if i==0 : ax.set_ylabel( y_labels[n], fontsize=30)
    if n == 0: ax.set_title( titles[i], fontsize=35)

# 
fig.suptitle(r'$\eta_0={0:0.3f}$   $\beta_0={1:0.3f}$   $\beta_1={2:0.3f}$'.format( eta_0, beta_0, beta_1 ), fontsize=30, y=0.997)
fig.tight_layout()
fileName = 'chemistry_{0}.png'.format(nSnap)
fig.savefig( outDir + fileName )
print 'Saved image: ', fileName
print ''
