import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib

# set some global options
matplotlib.font_manager.findSystemFonts(fontpaths=['/home/bruno/Downloads'], fontext='ttf')
# plt.rcParams['figure.figsize'] = (6,5)
# plt.rcParams['legend.frameon'] = False
# plt.rcParams['legend.fontsize'] = 14
# plt.rcParams['legend.borderpad'] = 0.1
# plt.rcParams['legend.labelspacing'] = 0.1
# plt.rcParams['legend.handletextpad'] = 0.1
plt.rcParams['font.family'] = 'Helvetica'


from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

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

dataDir = '/home/bruno/Desktop/data/'
# dataDir = '/raid/bruno/data/'

eta_1 = 0.001
eta_2 = 0.040
# beta_0 = 0.25
# beta_1 = 0.00

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

data_name = 'SIMPLE_PPMP_eta0.035_beta0.00_grav4'
chollaDir = dataDir + 'cosmo_sims/cholla_pm/{0}_cool_uv_50Mpc/'.format(nPoints)
chollaDir_uv = chollaDir +  'data_{0}/'.format( data_name )

enzoDir = dataDir + 'cosmo_sims/enzo/'
enzoDir_uv = enzoDir + '{0}_cool_uv_50Mpc_HLLC_grav4/h5_files/'.format(nPoints)


outDir = dev_dir + 'figures/projection_hydro_50Mpc/256_cool_{0}/'.format( data_name )
if rank == 0:
  create_directory( outDir )


metals = True

gamma = 5./3
nx = nPoints
ny = nPoints
nz = nPoints

dv = (Lbox/nPoints)**3

dens_weight = True

t_min, d_min = 1e20, 1e20
t_max, d_max = -1e20, -1e20

def get_projection( data, offset, depth, log=True ):
  if log: proj = np.log10(data[offset:offset+depth, :, :].sum(axis=0) )
  else: proj = data[offset:offset+depth, :, :].sum(axis=0)
  return proj


proj_offset = 0
proj_depth = 100

# fields = ['density_dm', 'density', 'HI_density',  'temperature' ]
fields = [ 'density_dm', 'density', 'HI_density', 'temperature' ]

ticks_list = [  [2, 5], [1, 4], [-5, -1.5],  [3.5, 7 ]]

cbar_labels = [ r'log Density  $[ h^2 \mathrm{M_{\odot} } \mathrm{kpc}^{-3}  ]$', r'log Density  $[ h^2 \mathrm{M_{\odot} } \mathrm{kpc}^{-3}  ]$',  r'log Density  $[ h^2 \mathrm{M_{\odot} } \mathrm{kpc}^{-3}  ]$',  r'log Temperature $[K ]$']

nSnap = 33
# n_snapshots = 10
# snapshots = range(0, n_snapshots)
# for nSnap in snapshots:
# 
data_ch = {}
data_cholla = load_snapshot_data( nSnap, chollaDir_uv, cool=True )
current_z_ch = data_cholla['current_z']
current_a_ch = data_cholla['current_a']
for i,field in enumerate(fields):
  # weight = data_cholla['gas'][field]
  # if field == 'temperature': weight = data_cholla['gas']['density']
  
  if field == 'density_dm': 
    data = data_cholla['dm']['density'][...]
    weight = data_cholla['dm']['density'][...]
  else:
    data = data_cholla['gas'][field][...]
    if field == 'HI_density': data[data>data.mean()*2000] = data.mean()*2000
    weight = data_cholla['gas']['density']
  data_weight = data * weight
  proj_data_wheight = get_projection( data_weight, proj_offset, proj_depth, log=False )
  proj_weight = get_projection( weight, proj_offset, proj_depth, log=False )
  proj = proj_data_wheight / proj_weight
  proj = np.log10( proj )
  # proj = get_projection( data, proj_offset, proj_depth ) 
  data_ch[field] = {}
  data_ch[field]['proj'] = proj
  data_ch[field]['max'] = proj.max()
  data_ch[field]['min'] = proj.min() 


data_en = {}
data_enzo = load_snapshot_enzo( nSnap, enzoDir_uv, dm=True, cool=True, metals=metals )
current_a_enzo = data_enzo['current_a']
current_z_enzo = data_enzo['current_z']
for i,field in enumerate(fields):
  if field == 'density_dm': 
    data = data_enzo['dm']['density'][...]
    weight = data_enzo['dm']['density'][...]
  
  else:
    data = data_enzo['gas'][field][...]
    if field == 'HI_density': data[data>data.mean()*2000] = data.mean()*2000
    weight = data_enzo['gas']['density']
  data_weight = data * weight
  proj_data_wheight = get_projection( data_weight, proj_offset, proj_depth, log=False )
  proj_weight = get_projection( weight, proj_offset, proj_depth, log=False )
  proj = proj_data_wheight / proj_weight
  # proj = get_projection( data, proj_offset, proj_depth ) 
  proj = np.log10( proj )
  data_en[field] = {}
  data_en[field]['proj'] = proj
  data_en[field]['max'] = proj.max()
  data_en[field]['min'] = proj.min() 

data_diff = {}
for i,field in enumerate(fields):
  if field=='density_dm':
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

n_rows = 2
n_cols = len(fields)
fig, ax_list = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(10*n_cols,9.2*n_rows))

plt.subplots_adjust(  wspace=0.5, hspace=0.3)
# titles = [ 'Z={0:.2f}   DM Density'.format(current_z_ch), 'Gas Density',  'HI', 'HII', 'Temperature' ]
titles = [ 'DM Density', 'Gas Density',  'HI Density', 'Temperature' ]
y_labels = [' ENZO', 'CHOLLA', 'DIFFERENCE' ]

# fig.suptitle(r'$\eta_1={0:0.3f}$   $\eta_2={1:0.4}$   '.format( eta_1, eta_2 ), fontsize=30, y=0.997)

for i in range( n_cols):
  field = fields[i]
  min_val = min( data_en[field]['min'], data_ch[field]['min'], )
  max_val = max( data_en[field]['max'], data_ch[field]['max'], )


  for n in range(n_rows):
    data = data_all[n]


    ax = ax_list[n][i]
    proj = data[field]['proj']

    # if n == n_rows-1:
    #   # min_val = max( -10, proj.min())
    #   # max_val = min( 10 , proj.max() )
    #   min_val = -1
    #   max_val = 3
    #   im = ax.imshow( proj, interpolation='bilinear',  vmin=min_val, vmax=max_val, cmap='jet' )

    # else:
    if field == 'density_dm': colormap = 'inferno'
    if field == 'density': colormap = 'cividis'
    if field == 'HI_density': colormap = 'cividis'
    if field == 'temperature': colormap = 'gist_heat'
    
    im = ax.imshow( proj, interpolation='bilinear',  vmin=min_val, vmax=max_val, cmap=colormap )
    

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = fig.colorbar( im, cax=cax, ticks=ticks_list[i] )
    cb.ax.tick_params(labelsize=23, size=7)
    cb_fs = 25
    if i ==0:cb.set_label(cbar_labels[i], fontsize=cb_fs,  labelpad=-8)
    if i ==1:cb.set_label(cbar_labels[i], fontsize=cb_fs,  labelpad=-8)
    if i ==2:cb.set_label(cbar_labels[i], fontsize=cb_fs,  labelpad=-40)
    if i ==3:cb.set_label(cbar_labels[i], fontsize=cb_fs,  labelpad=-25)

    ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    if i==0 : ax.set_ylabel( y_labels[n], fontsize=40)
    if n == 0: ax.set_title( titles[i], fontsize=35)

# 
fig.tight_layout()
fileName = 'projection_{0}_cividis.png'.format(nSnap)
fig.savefig( outDir + fileName,  bbox_inches='tight', dpi=100 )
print 'Saved image: ', fileName
print ''
