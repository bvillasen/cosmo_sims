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

eta_1 = 0.001
eta_2 = 0.030



n_arg = len(sys.argv)
if n_arg > 1:
  args = []
  for i in range(1 , n_arg):
    arg = sys.argv[i]
    args.append( float( arg ))
  eta_1, eta_2 = args
  if rank == 0:
    print "Using command arguments"
    print args


print 'eta: {0:.3f}  {1:.3f} /'.format( eta_1, eta_2 )


nPoints = 256
Lbox = 50000.

integrator = 'SIMPLE'

chollaDir = dataDir + 'cosmo_sims/cholla_pm/{0}_cool_uv_50Mpc/'.format(nPoints)
chollaDir_uv = chollaDir +  'data_PPMC_HLLC_{2}_eta{0:.3f}_{1:.3f}/'.format( eta_1, eta_2, integrator )

enzoDir = dataDir + 'cosmo_sims/enzo/'
enzoDir_uv = enzoDir + '{0}_cool_uv_50Mpc/h5_files/'.format(nPoints)


outDir = dev_dir + 'figures/chemistry/chemistry_HI_256_PPMC_eta{0:.3f}_{1:.3f}/'.format( eta_1, eta_2 )
if rank == 0:
  create_directory( outDir )


metals = True

gamma = 5./3
nx = nPoints
ny = nPoints
nz = nPoints

dv = (Lbox/nPoints)**3

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
proj_depth = 64

fields = ['density_dm', 'density', 'HI_density', 'HII_density', 'temperature' ]

# nSnap = 0
# n_snapshots = 10
# snapshots = range(0, n_snapshots)
# for nSnap in snapshots:
# 
data_ch = {}
data_cholla = load_snapshot_data( nSnap, chollaDir_uv, cool=True )
current_z_ch = data_cholla['current_z']
current_a_ch = data_cholla['current_a']
for i,field in enumerate(fields):
  if i == 0:
   data = data_cholla['dm']['density'][...]
  else:
    data = data_cholla['gas'][field][...]
  data_ch[field] = {}
  proj = get_projection( data, proj_offset, proj_depth )
  data_ch[field]['proj'] = proj
  data_ch[field]['max'] = proj.max()
  data_ch[field]['min'] = proj.min() 


data_en = {}
data_enzo = load_snapshot_enzo( nSnap, enzoDir_uv, dm=True, cool=True, metals=metals )
current_a_enzo = data_enzo['current_a']
current_z_enzo = data_enzo['current_z']
for i,field in enumerate(fields):
  if i == 0:
   data = data_enzo['dm']['density'][...]
  else:
    data = data_enzo['gas'][field][...]
  data_en[field] = {}
  proj = get_projection( data, proj_offset, proj_depth )
  data_en[field]['proj'] = proj
  data_en[field]['max'] = proj.max()
  data_en[field]['min'] = proj.min() 

data_diff = {}
for i,field in enumerate(fields):
  if i == 0: 
    vals_ch = data_cholla['dm']['density'][...] 
    vals_en = data_enzo['dm']['density'][...]
  else: 
    vals_ch = data_cholla['gas'][field][...] 
    vals_en = data_enzo['gas'][field][...]
  # if i == 0: print vals_ch.mean()
  proj_ch = get_projection( vals_ch, proj_offset, proj_depth, log=False )
  proj_en = get_projection( vals_en, proj_offset, proj_depth, log=False )
  proj = ( proj_ch - proj_en )/ proj_en
  data_diff[field] = {}
  data_diff[field]['proj'] = proj
  data_diff[field]['max'] = proj.max()
  data_diff[field]['min'] = proj.min() 



data_all = [ data_en, data_ch, data_diff ]

n_rows = 3
n_cols = len(fields)
fig, ax_list = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(10*n_cols,10*n_rows))
titles = [ 'Z={0:.2f}   DM Density'.format(current_z_ch), 'Gas Density',  'HI', 'HII', 'Temperature' ]
y_labels = [' ENZO', 'CHOLLA', 'DIFFERENCE' ]

fig.suptitle(r'$\eta_1={0:0.3f}$   $\eta_2={1:0.4}$   '.format( eta_1, eta_2 ), fontsize=30, y=0.997)

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
      min_val = -1
      max_val = 1
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
fig.tight_layout()
fileName = 'chemistry_{0}.png'.format(nSnap)
fig.savefig( outDir + fileName )
print 'Saved image: ', fileName
print ''
