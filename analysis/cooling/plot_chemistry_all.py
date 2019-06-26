import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
from mpl_toolkits.axes_grid1 import make_axes_locatable

cosmo_dir = '/home/bruno/Desktop/Dropbox/Developer/cosmo_sims/'
outDir = cosmo_dir + 'figures/power_hydro/'
toolsDirectory = cosmo_dir + "tools/"
sys.path.extend([toolsDirectory ] )
from tools import create_directory
from load_data_cholla import load_snapshot_data
from load_data_enzo import load_snapshot_enzo


# dataDir = '/home/bruno/Desktop/data/'
dataDir = '/raid/bruno/data/'

chollaDir = dataDir + 'cosmo_sims/cholla_pm/256_cool/'
chollaDir_uv = chollaDir +  'data/'
chollaDir_metals = chollaDir +  'data_metals/'

enzoDir = dataDir + 'cosmo_sims/enzo/'
enzoDir_uv = enzoDir + '256_cool_uv/h5_files/'
enzoDir_metals = enzoDir + '256_hydro_grackle_noUV/h5_files/'



outDir = cosmo_dir + 'figures/cooling/chemistry_new/'
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

nSnap = 1
n_snapshots = 10
snapshots = range(0, n_snapshots)
for nSnap in snapshots:

  data = {}

  data_cholla = load_snapshot_data( nSnap, chollaDir_uv, cool=True )
  current_z_ch = data_cholla['current_z']
  current_a_ch = data_cholla['current_a']
  print current_z_ch
  dens_ch = data_cholla['gas']['density'][...]
  temp_ch = data_cholla['gas']['temperature'][...]
  gasEnergy_ch = data_cholla['gas']['GasEnergy'][...] /dens_ch
  dens_avrg = dens_ch.mean()
  HI_dens_ch = data_cholla['gas']['HI_density'][...]
  HII_dens_ch = data_cholla['gas']['HII_density'][...]
  HeI_dens_ch = data_cholla['gas']['HeI_density'][...]
  HeII_dens_ch = data_cholla['gas']['HeII_density'][...]
  HeIII_dens_ch = data_cholla['gas']['HeIII_density'][...]
  e_dens_ch = data_cholla['gas']['e_density'][...]
  metal_dens_ch = data_cholla['gas']['metal_density'][...]
  dens_proj_ch = np.log10(dens_ch[:n_slice, :, :].sum(axis=0))
  HI_dens_proj_ch = np.log10(HI_dens_ch[:n_slice, :, :].sum(axis=0))
  HII_dens_proj_ch = np.log10(HII_dens_ch[:n_slice, :, :].sum(axis=0))
  HeI_dens_proj_ch = np.log10(HeI_dens_ch[:n_slice, :, :].sum(axis=0))
  HeII_dens_proj_ch = np.log10(HeII_dens_ch[:n_slice, :, :].sum(axis=0))
  HeIII_dens_proj_ch = np.log10(HeIII_dens_ch[:n_slice, :, :].sum(axis=0))
  e_dens_proj_ch = np.log10(e_dens_ch[:n_slice, :, :].sum(axis=0))
  gasEnergy_proj_ch = np.log10(gasEnergy_ch[:n_slice, :, :].sum(axis=0))
  metal_dens_proj_ch = np.log10(metal_dens_ch[:n_slice, :, :].sum(axis=0))

  m = 0.9718
  data['cholla_uv'] = {}
  data['cholla_uv'][0] =  dens_proj_ch
  data['cholla_uv'][1] =  HI_dens_proj_ch
  data['cholla_uv'][2] =  HII_dens_proj_ch 
  data['cholla_uv'][3] =  HeI_dens_proj_ch
  data['cholla_uv'][4] =  HeII_dens_proj_ch 
  data['cholla_uv'][5] =  HeIII_dens_proj_ch
  data['cholla_uv'][6] =  e_dens_proj_ch 
  data['cholla_uv'][7] =  metal_dens_proj_ch
  data['cholla_uv'][8] = gasEnergy_proj_ch 

  # data_cholla = load_snapshot_data( nSnap, chollaDir_metals, cool=True )
  # current_z_ch = data_cholla['current_z']
  # current_a_ch = data_cholla['current_a']
  # print current_z_ch
  # dens_ch = data_cholla['gas']['density'][...]
  # temp_ch = data_cholla['gas']['temperature'][...]
  # gasEnergy_ch = data_cholla['gas']['GasEnergy'][...] /dens_ch
  # dens_avrg = dens_ch.mean()
  # HI_dens_ch = data_cholla['gas']['HI_density'][...]
  # HII_dens_ch = data_cholla['gas']['HII_density'][...]
  # HeI_dens_ch = data_cholla['gas']['HeI_density'][...]
  # HeII_dens_ch = data_cholla['gas']['HeII_density'][...]
  # HeIII_dens_ch = data_cholla['gas']['HeIII_density'][...]
  # e_dens_ch = data_cholla['gas']['e_density'][...]
  # metal_dens_ch = data_cholla['gas']['metal_density'][...]
  # dens_proj_ch = np.log10(dens_ch[:n_slice, :, :].sum(axis=0))
  # HI_dens_proj_ch = np.log10(HI_dens_ch[:n_slice, :, :].sum(axis=0))
  # HII_dens_proj_ch = np.log10(HII_dens_ch[:n_slice, :, :].sum(axis=0))
  # HeI_dens_proj_ch = np.log10(HeI_dens_ch[:n_slice, :, :].sum(axis=0))
  # HeII_dens_proj_ch = np.log10(HeII_dens_ch[:n_slice, :, :].sum(axis=0))
  # HeIII_dens_proj_ch = np.log10(HeIII_dens_ch[:n_slice, :, :].sum(axis=0))
  # e_dens_proj_ch = np.log10(e_dens_ch[:n_slice, :, :].sum(axis=0))
  # gasEnergy_proj_ch = np.log10(gasEnergy_ch[:n_slice, :, :].sum(axis=0))
  # # if nSnap == 1: gasEnergy_ch *= 1.1
  # metal_dens_proj_ch = np.log10(metal_dens_ch[:n_slice, :, :].sum(axis=0))
  #
  #
  # data['cholla_metals'] = {}
  # data['cholla_metals'][0] =  dens_proj_ch
  # data['cholla_metals'][1] =  HI_dens_proj_ch
  # data['cholla_metals'][2] =  HII_dens_proj_ch
  # data['cholla_metals'][3] =  HeI_dens_proj_ch
  # data['cholla_metals'][4] =  HeII_dens_proj_ch
  # data['cholla_metals'][5] =  HeIII_dens_proj_ch
  # data['cholla_metals'][6] =  e_dens_proj_ch
  # data['cholla_metals'][7] =  metal_dens_proj_ch
  # data['cholla_metals'][8] = gasEnergy_proj_ch
  # if nSnap == 1 : data['cholla_metals'][8] = data['cholla_uv'][8]



  print ' Enzo: ', nSnap
  data_enzo = load_snapshot_enzo( nSnap, enzoDir_uv, cool=True, metals=metals )
  current_a_enzo = data_enzo['current_a']
  current_z_enzo = data_enzo['current_z']
  dens_en = data_enzo['gas']['density'][...]
  temp_en = data_enzo['gas']['temperature'][...]
  gasEnergy_en = data_enzo['gas']['GasEnergy'][...] /dens_en
  dens_avrg = dens_en.mean()
  HI_dens_en = data_enzo['gas']['HI_density'][...]
  HII_dens_en = data_enzo['gas']['HII_density'][...]
  HeI_dens_en = data_enzo['gas']['HeI_density'][...]
  HeII_dens_en = data_enzo['gas']['HeII_density'][...]
  HeIII_dens_en = data_enzo['gas']['HeIII_density'][...]
  e_dens_en = data_enzo['gas']['e_density'][...]
  metal_dens_en = data_enzo['gas']['metal_density'][...]
  proten_electron_mass_ratio = 1836.15267389
  e_dens_en *= proten_electron_mass_ratio
  dens_proj_en = np.log10(dens_en[:n_slice, :, :].sum(axis=0))
  HI_dens_proj_en = np.log10(HI_dens_en[:n_slice, :, :].sum(axis=0))
  HII_dens_proj_en = np.log10(HII_dens_en[:n_slice, :, :].sum(axis=0))
  HeI_dens_proj_en = np.log10(HeI_dens_en[:n_slice, :, :].sum(axis=0))
  HeII_dens_proj_en = np.log10(HeII_dens_en[:n_slice, :, :].sum(axis=0))
  HeIII_dens_proj_en = np.log10(HeIII_dens_en[:n_slice, :, :].sum(axis=0))
  e_dens_proj_en = np.log10(e_dens_en[:n_slice, :, :].sum(axis=0))
  gasEnergy_proj_en = np.log10(gasEnergy_en[:n_slice, :, :].sum(axis=0))
  metal_dens_proj_en = np.log10(metal_dens_en[:n_slice, :, :].sum(axis=0))

  data['enzo_uv'] = {}
  data['enzo_uv'][0] =  dens_proj_en
  data['enzo_uv'][1] =  HI_dens_proj_en
  data['enzo_uv'][2] =  HII_dens_proj_en
  data['enzo_uv'][3] =  HeI_dens_proj_en
  data['enzo_uv'][4] =  HeII_dens_proj_en
  data['enzo_uv'][5] =  HeIII_dens_proj_en
  data['enzo_uv'][6] =  e_dens_proj_en
  data['enzo_uv'][7] =  metal_dens_proj_en
  data['enzo_uv'][8] = gasEnergy_proj_en
  #
  # print ' Enzo: ', nSnap
  # data_enzo = load_snapshot_enzo( nSnap, enzoDir_metals, cool=True, metals=metals )
  # current_a_enzo = data_enzo['current_a']
  # current_z_enzo = data_enzo['current_z']
  # dens_en = data_enzo['gas']['density'][...]
  # temp_en = data_enzo['gas']['temperature'][...]
  # gasEnergy_en = data_enzo['gas']['GasEnergy'][...] /dens_en
  # dens_avrg = dens_en.mean()
  # HI_dens_en = data_enzo['gas']['HI_density'][...]
  # HII_dens_en = data_enzo['gas']['HII_density'][...]
  # HeI_dens_en = data_enzo['gas']['HeI_density'][...]
  # HeII_dens_en = data_enzo['gas']['HeII_density'][...]
  # HeIII_dens_en = data_enzo['gas']['HeIII_density'][...]
  # e_dens_en = data_enzo['gas']['e_density'][...]
  # metal_dens_en = data_enzo['gas']['metal_density'][...]
  # proten_electron_mass_ratio = 1836.15267389
  # e_dens_en *= proten_electron_mass_ratio
  # dens_proj_en = np.log10(dens_en[:n_slice, :, :].sum(axis=0))
  # HI_dens_proj_en = np.log10(HI_dens_en[:n_slice, :, :].sum(axis=0))
  # HII_dens_proj_en = np.log10(HII_dens_en[:n_slice, :, :].sum(axis=0))
  # HeI_dens_proj_en = np.log10(HeI_dens_en[:n_slice, :, :].sum(axis=0))
  # HeII_dens_proj_en = np.log10(HeII_dens_en[:n_slice, :, :].sum(axis=0))
  # HeIII_dens_proj_en = np.log10(HeIII_dens_en[:n_slice, :, :].sum(axis=0))
  # e_dens_proj_en = np.log10(e_dens_en[:n_slice, :, :].sum(axis=0))
  # gasEnergy_proj_en = np.log10(gasEnergy_en[:n_slice, :, :].sum(axis=0))
  # metal_dens_proj_en = np.log10(metal_dens_en[:n_slice, :, :].sum(axis=0))
  #
  # data['enzo_metals'] = {}
  # data['enzo_metals'][0] =  dens_proj_en
  # data['enzo_metals'][1] =  HI_dens_proj_en
  # data['enzo_metals'][2] =  HII_dens_proj_en
  # data['enzo_metals'][3] =  HeI_dens_proj_en
  # data['enzo_metals'][4] =  HeII_dens_proj_en
  # data['enzo_metals'][5] =  HeIII_dens_proj_en
  # data['enzo_metals'][6] =  e_dens_proj_en
  # data['enzo_metals'][7] =  metal_dens_proj_en
  # data['enzo_metals'][8] = gasEnergy_proj_en


  data['max_vals'] = {}
  data['min_vals'] = {}
  # types = [ 'enzo_uv', 'cholla_uv', 'enzo_metals', 'cholla_metals']
  types = ['enzo_uv', 'cholla_uv']
  for i in range(9):
    max_val, min_val = -1e100, 1e100
    for type in types:
      max_val = max( max_val, data[type][i].max() )
      min_val = min( max_val, data[type][i].min() )
    data['max_vals'][i] = max_val
    data['min_vals'][i] = min_val

  n_rows = len(types)
  fig, ax_list = plt.subplots(nrows=n_rows, ncols=9, figsize=(90,10*n_rows))
  titles = [ 'Z={0}   All'.format(current_z_ch), 'HI', 'HII', "HeI", 'HeII', 'HeIII', 'electron', 'metals', 'temperature' ]
  y_labels = [' ENZO UVB', 'CHOLLA UVB', 'ENZO noUVB', 'CHOLLA_noUVB']

  for row in range(n_rows):
    type = types[row]
    for i in range( 9):
      ax = ax_list[row][i]
      if i == 8: im = ax.imshow( data[type][i], interpolation='bilinear', vmin=data['min_vals'][i], vmax=data['max_vals'][i], cmap='jet' )
      else: im = ax.imshow( data[type][i], interpolation='bilinear', vmin=data['min_vals'][i], vmax=data['max_vals'][i] )
      divider = make_axes_locatable(ax)
      cax = divider.append_axes("right", size="5%", pad=0.05)
      fig.colorbar( im, cax=cax )
      ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
      if i==0 : ax.set_ylabel( y_labels[row], fontsize=30)
      if row == 0: ax.set_title( titles[i], fontsize=35)


  fig.tight_layout()
  fileName = 'chemistry_{0}.png'.format(nSnap)
  fig.savefig( outDir + fileName )
  print 'Saved image: ', fileName
  print ''
