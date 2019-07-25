import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5

dev_dir = '/home/bruno/Desktop/Dropbox/Developer/'
cosmo_dir = dev_dir + 'cosmo_sims/'
toolsDirectory = cosmo_dir + "tools/"
sys.path.extend([toolsDirectory ] )
from load_data_cholla import load_snapshot_data
from load_data_enzo import load_snapshot_enzo
from phase_diagram import get_phase_diagram
from internal_energy import get_internal_energy, get_temp, get_Temperaure_From_Flags_DE
from tools import create_directory

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nSnap = rank

# name = 'de001_PLMP_CTU'
dataDir = '/raid/bruno/data/'


integrator = 'SIMPLE'
extra_name = '_pressure'

outDir = dev_dir + 'figures/cell_difference/'
create_directory( outDir )
print "Output: ", outDir


Lbox = 50000.
nPoints = 128 
nCells = nPoints**3


enzoDir_uv = dataDir + 'cosmo_sims/enzo/{0}_cool_uv/h5_files/'.format(nPoints)


chollaDir_base = dataDir + 'cosmo_sims/cholla_pm/{0}_cool/'.format(nPoints)

eta_1 = 0.001
eta_2_LIST = [ 0.020, 0.022, 0.024, 0.026, 0.028,  0.030, 0.032, 0.034, 0.036, 0.038, 0.040, 0.042, 0.044, 0.046, 0.048, 0.050  ]

n_eta_vals = len(eta_2_LIST)

nSnap = 0

for nSnap in range(0, 28):
  fileName = 'l1_cell_difference_{0}.png'.format(nSnap)



  field_list = [ 'density', 'density', 'HI_density', 'temperature' ]
  title_list = ['DM Density', "Gas Density", 'HI Density', 'Temperature']
  diff_list_all = []

  n_fields = len( field_list)

  for n_eta in range(n_eta_vals):
    eta_2 = eta_2_LIST[n_eta]
    chollaDir = chollaDir_base +  'data_PPMC_HLLC_{2}_eta{0:.3f}_{1:.3f}{3}/'.format( eta_1, eta_2, integrator, extra_name )

    data_cholla = load_snapshot_data( str(nSnap), chollaDir, cool=True)
    current_z_ch = data_cholla['current_z']
    current_a_ch = data_cholla['current_a']

    data_enzo = load_snapshot_enzo( nSnap, enzoDir_uv, dm=True, cool=True)
    current_z_en = data_enzo['current_z']


    diff_list = []
    for n_field in range(n_fields):
      field = field_list[n_field]
      
      if n_field == 0: type = 'dm'
      else: type = 'gas'
      vals_ch = data_cholla[type][field][...]
      vals_en = data_enzo[type][field][...]
      if type=='dm': 
        dens_mean_ch = vals_ch.mean()
        dens_mean_en = vals_en.mean()
        print dens_mean_en / dens_mean_ch 
        vals_ch[vals_en<1e-5] = 1e-6*dens_mean_ch
        vals_en[vals_en<1e-5] = 1e-6*dens_mean_ch
      diff = ( vals_ch - vals_en ) / vals_en
      l1_diff = np.sum( np.abs(diff) ) / nCells
      
      if field == 'temperature' and current_z_ch>15: l1_diff /= 5
      diff_list.append( l1_diff)
      
      if type=='dm':
        print l1_diff
      
      
    diff_list_all.append(diff_list)
    
  diff_list_all = np.array( diff_list_all).T
    
  nrows = 1  
  ncols = n_fields
  fig, ax_l = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10*ncols,8*nrows))
  x_min = 0.018
  x_max = 0.052

  y_min_list = [ 0,  0 , 0, 0 ]
  y_max_list = [ 0.5, 0.1, 0.5, 0.7 ]

  for n in range(n_fields):
    
    diff_vals = diff_list_all[n]
    
    plt.subplot(nrows, ncols, n+1)
    ax = plt.gca()
    ax.clear()
    plt.plot( eta_2_LIST, diff_vals, 'o' )
    # c = ax.scatter( data[1], data[0], c = np.log10(data[2]), s=1, vmin=min_val, vmax=max_val )
    # plt.colorbar(c)
    ax.set_ylabel(r'L1 Difference', fontsize=16 )
    ax.set_xlabel(r'$\eta_2$', fontsize=16 )
    ax.set_title( title_list[n], fontsize=17 )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min_list[n], y_max_list[n] )
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=8)



  fig.suptitle(r'Z={0:.2f}    L1 Cell Difference'.format( current_z_en ), fontsize=20, y=0.999)



  # fig.tight_layout()
  fig.savefig( outDir + fileName,  pad_inches=0.1,  bbox_inches='tight',)
  print 'Saved image: ', fileName
  print ''