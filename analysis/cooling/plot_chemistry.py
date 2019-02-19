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


# dataDir = '/home/bruno/Desktop/data/'
dataDir = '/raid/bruno/data/'

chollaDir = dataDir + 'cosmo_sims/cholla_pm/256_cool/'
chollaDir_cool = chollaDir +  'data/'
chollaDir_0 = chollaDir +  'data_hydro/'

enzoDir = dataDir + 'cosmo_sims/enzo/'
enzoDir_cool = enzoDir + '256_hydro_grackle/h5_files/'
enzoDir_0 = enzoDir + '256_hydro/h5_files/'


outDir = cosmo_dir + 'figures/cooling/chemistry/'

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

n_snapshots = 31
snapshots = range(0, n_snapshots)
nSnap = 0
for nSnap in snapshots:
  snapKey = str( nSnap )

  # data_cholla = load_snapshot_data( str(nSnap), chollaDir_0, cool=True )
  # current_z_ch = data_cholla['current_z']
  # current_a_ch = data_cholla['current_a']
  # dens_ch_0 = data_cholla['gas']['density'][...]
  # gasEnergy_ch_0 = data_cholla['gas']['GasEnergy'][...] /dens_ch_0
  # dens_avrg = dens_ch_0.mean()
  # dens_ch_0 /= dens_avrg
  # dens_proj_ch_0 = np.log10(dens_ch_0[:n_slice, :, :].sum(axis=0))
  # gasEnergy_proj_ch_0 = np.log10(gasEnergy_ch_0[:n_slice, :, :].sum(axis=0))

  data_cholla = load_snapshot_data( str(nSnap), chollaDir_cool, cool=True )
  current_z_ch = data_cholla['current_z']
  current_a_ch = data_cholla['current_a']
  print current_z_ch
  dens_ch = data_cholla['gas']['density'][...]
  temp_ch = data_cholla['gas']['temperature'][...]
  gasEnergy_ch = data_cholla['gas']['GasEnergy'][...] /dens_ch
  dens_avrg = dens_ch.mean()
  # dens_ch /= dens_avrg
  # HI_dens_ch = data_cholla['gas']['HI_density'][...] / dens_avrg
  # HII_dens_ch = data_cholla['gas']['HII_density'][...] / dens_avrg
  # HeI_dens_ch = data_cholla['gas']['HeI_density'][...] / dens_avrg
  # HeII_dens_ch = data_cholla['gas']['HeII_density'][...] / dens_avrg
  # HeIII_dens_ch = data_cholla['gas']['HeIII_density'][...] / dens_avrg
  # e_dens_ch = data_cholla['gas']['e_density'][...] / dens_avrg

  HI_dens_ch = data_cholla['gas']['HI_density'][...]
  HII_dens_ch = data_cholla['gas']['HII_density'][...]
  HeI_dens_ch = data_cholla['gas']['HeI_density'][...]
  HeII_dens_ch = data_cholla['gas']['HeII_density'][...]
  HeIII_dens_ch = data_cholla['gas']['HeIII_density'][...]
  e_dens_ch = data_cholla['gas']['e_density'][...]
  dens_proj_ch = np.log10(dens_ch[:n_slice, :, :].sum(axis=0))
  HI_dens_proj_ch = np.log10(HI_dens_ch[:n_slice, :, :].sum(axis=0))
  HII_dens_proj_ch = np.log10(HII_dens_ch[:n_slice, :, :].sum(axis=0))
  HeI_dens_proj_ch = np.log10(HeI_dens_ch[:n_slice, :, :].sum(axis=0))
  HeII_dens_proj_ch = np.log10(HeII_dens_ch[:n_slice, :, :].sum(axis=0))
  HeIII_dens_proj_ch = np.log10(HeIII_dens_ch[:n_slice, :, :].sum(axis=0))
  e_dens_proj_ch = np.log10(e_dens_ch[:n_slice, :, :].sum(axis=0))
  gasEnergy_proj_ch = np.log10(gasEnergy_ch[:n_slice, :, :].sum(axis=0))


  dens_max_ch = dens_proj_ch.max()
  HI_max_ch = HI_dens_proj_ch.max()
  HII_max_ch = HII_dens_proj_ch.max()
  HeI_max_ch = HeI_dens_proj_ch.max()
  HeII_max_ch = HeII_dens_proj_ch.max()
  HeIII_max_ch = HeIII_dens_proj_ch.max()
  e_max_ch = e_dens_proj_ch.max()


  dens_min_ch = dens_proj_ch.min()
  HI_min_ch = HI_dens_proj_ch.min()
  HII_min_ch = HII_dens_proj_ch.min()
  HeI_min_ch = HeI_dens_proj_ch.min()
  HeII_min_ch = HeII_dens_proj_ch.min()
  HeIII_min_ch = HeIII_dens_proj_ch.min()
  e_min_ch = e_dens_proj_ch.min()

  if metals:
    metal_dens_ch = data_cholla['gas']['metal_density'][...]
    metal_dens_proj_ch = np.log10(metal_dens_ch[:n_slice, :, :].sum(axis=0))
    metal_max_ch = metal_dens_proj_ch.max()
    metal_min_ch = metal_dens_proj_ch.min()


  # data_enzo = load_snapshot_enzo( nSnap, enzoDir_0 )
  # current_a_enzo = data_enzo['current_a']
  # current_z_enzo = data_enzo['current_z']
  # dens_en_0 = data_enzo['gas']['density'][...]
  # gasEnergy_en_0 = data_enzo['gas']['GasEnergy'][...] /dens_en_0
  # dens_avrg = dens_en_0.mean()
  # dens_en_0 /= dens_avrg
  # dens_proj_en_0 = np.log10(dens_en_0[:n_slice, :, :].sum(axis=0))
  # gasEnergy_proj_en_0 = np.log10(gasEnergy_en_0[:n_slice, :, :].sum(axis=0))


  print ' Enzo: ', nSnap
  data_enzo = load_snapshot_enzo( nSnap, enzoDir_cool, cool=True, metals=metals )
  current_a_enzo = data_enzo['current_a']
  current_z_enzo = data_enzo['current_z']
  dens_en = data_enzo['gas']['density'][...]
  temp_en = data_enzo['gas']['temperature'][...]
  gasEnergy_en = data_enzo['gas']['GasEnergy'][...] /dens_en
  dens_avrg = dens_en.mean()
  # dens_en /= dens_avrg
  # HI_dens_en = data_enzo['gas']['HI_density'][...] / dens_avrg
  # HII_dens_en = data_enzo['gas']['HII_density'][...] / dens_avrg
  # HeI_dens_en = data_enzo['gas']['HeI_density'][...] / dens_avrg
  # HeII_dens_en = data_enzo['gas']['HeII_density'][...] / dens_avrg
  # HeIII_dens_en = data_enzo['gas']['HeIII_density'][...] / dens_avrg
  # e_dens_en = data_enzo['gas']['e_density'][...] / dens_avrg
  HI_dens_en = data_enzo['gas']['HI_density'][...]
  HII_dens_en = data_enzo['gas']['HII_density'][...]
  HeI_dens_en = data_enzo['gas']['HeI_density'][...]
  HeII_dens_en = data_enzo['gas']['HeII_density'][...]
  HeIII_dens_en = data_enzo['gas']['HeIII_density'][...]
  e_dens_en = data_enzo['gas']['e_density'][...]
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

  dens_max_en = dens_proj_en.max()
  HI_max_en = HI_dens_proj_en.max()
  HII_max_en = HII_dens_proj_en.max()
  HeI_max_en = HeI_dens_proj_en.max()
  HeII_max_en = HeII_dens_proj_en.max()
  HeIII_max_en = HeIII_dens_proj_en.max()
  e_max_en = e_dens_proj_en.max()


  dens_min_en = dens_proj_en.min()
  HI_min_en = HI_dens_proj_en.min()
  HII_min_en = HII_dens_proj_en.min()
  HeI_min_en = HeI_dens_proj_en.min()
  HeII_min_en = HeII_dens_proj_en.min()
  HeIII_min_en = HeIII_dens_proj_en.min()
  e_min_en = e_dens_proj_en.min()

  if metals:
    metal_dens_en = data_enzo['gas']['metal_density'][...]
    metal_dens_proj_en = np.log10(metal_dens_en[:n_slice, :, :].sum(axis=0))
    metal_min_en = metal_dens_proj_en.min()
    metal_max_en = metal_dens_proj_en.max()


  HI_max = max( dens_max_ch, dens_max_en)
  HI_max = max( HI_max_ch, HI_max_en)
  HII_max = max( HII_max_ch, HII_max_en)
  HeI_max = max( HeI_max_ch, HeI_max_en)
  HeII_max = max( HeII_max_ch, HeII_max_en)
  HeIII_max = max( HeIII_max_ch, HeIII_max_en)
  e_max = max( e_max_ch, e_max_en)


  dens_min = min( dens_min_ch, dens_min_en)
  HI_min = min( HI_min_ch, HI_min_en)
  HII_min = min( HII_min_ch, HII_min_en)
  HeI_min = min( HeI_min_ch, HeI_min_en)
  HeII_min = min( HeII_min_ch, HeII_min_en)
  HeIII_min = min( HeIII_min_ch, HeIII_min_en)
  e_min = min( e_min_ch, e_min_en)

  if metals:
    metal_max = max( metal_max_ch, metal_max_en)
    metal_min = min( metal_min_ch, metal_min_en)

  ge_all_max = [ gasEnergy_proj_ch.max(), gasEnergy_proj_en.max() ]
  ge_all_min = [ gasEnergy_proj_ch.min(), gasEnergy_proj_en.min() ]
  # ge_all_max = [ gasEnergy_proj_ch.max(), gasEnergy_proj_en.max(), gasEnergy_proj_ch_0.max(), gasEnergy_proj_en_0.max() ]
  # ge_all_min = [ gasEnergy_proj_ch.min(), gasEnergy_proj_en.min(), gasEnergy_proj_ch_0.min(), gasEnergy_proj_en_0.min() ]
  ge_max = max( ge_all_max )
  ge_min = min( ge_all_min )


  fig, ax_list = plt.subplots(nrows=2, ncols=8, figsize=(80,20))
  if metals:
    fig, ax_list = plt.subplots(nrows=2, ncols=9, figsize=(90,20))

  ax = ax_list[0][0]
  im = ax.imshow( dens_proj_en, interpolation='bilinear', )
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="5%", pad=0.05)
  fig.colorbar( im, cax=cax )
  ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
  ax.set_ylabel( " ENZO Overdensity ", fontsize=30)
  ax.set_title( "Z={0:.02f}  All".format(current_z_ch), fontsize=30)

  ax = ax_list[0][1]
  im = ax.imshow( HI_dens_proj_en, interpolation='bilinear', vmin=HI_min, vmax=HI_max)
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="5%", pad=0.05)
  fig.colorbar( im, cax=cax )
  ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
  ax.set_title( "HI".format(current_z_ch), fontsize=30)
  #
  ax = ax_list[0][2]
  im = ax.imshow( HII_dens_proj_en, interpolation='bilinear', vmin=HII_min, vmax=HII_max)
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="5%", pad=0.05)
  fig.colorbar( im, cax=cax )
  ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
  ax.set_title( "HII".format(current_z_ch), fontsize=30)
  #
  ax = ax_list[0][3]
  im = ax.imshow( HeI_dens_proj_en, interpolation='bilinear', vmin=HeI_min, vmax=HeI_max)
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="5%", pad=0.05)
  fig.colorbar( im, cax=cax )
  ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
  ax.set_title( "HeI".format(current_z_ch), fontsize=30)
  #

  ax = ax_list[0][4]
  im = ax.imshow( HeII_dens_proj_en, interpolation='bilinear', vmin=HeII_min, vmax=HeII_max)
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="5%", pad=0.05)
  fig.colorbar( im, cax=cax )
  ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
  ax.set_title( "HeII".format(current_z_ch), fontsize=30)
  #
  #
  #
  #
  ax = ax_list[0][5]
  im = ax.imshow( HeIII_dens_proj_en, interpolation='bilinear', vmin=HeIII_min, vmax=HeIII_max)
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="5%", pad=0.05)
  fig.colorbar( im, cax=cax )
  ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
  ax.set_title( "HeIII".format(current_z_ch), fontsize=30)
  #
  #
  ax = ax_list[0][6]
  im = ax.imshow( e_dens_proj_en, interpolation='bilinear', vmin=e_min, vmax=e_max)
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="5%", pad=0.05)
  fig.colorbar( im, cax=cax )
  ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
  ax.set_title( "electron".format(current_z_ch), fontsize=30)


  ax = ax_list[0][7]
  im = ax.imshow( gasEnergy_proj_en, interpolation='bilinear', cmap='jet', vmin=ge_min, vmax=ge_max)
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="5%", pad=0.05)
  fig.colorbar( im, cax=cax )
  ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
  ax.set_title( "Temperature".format(current_z_ch), fontsize=30)

  if metals:
    ax = ax_list[0][8]
    im = ax.imshow( metal_dens_proj_en, interpolation='bilinear', vmin=metal_min, vmax=metal_max)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar( im, cax=cax )
    ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    ax.set_title( "metals".format(current_z_ch), fontsize=30)

  ax = ax_list[1][0]
  im = ax.imshow( dens_proj_ch, interpolation='bilinear', )
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="5%", pad=0.05)
  fig.colorbar( im, cax=cax )
  ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
  ax.set_ylabel( "  CHOLLA Overdensity ", fontsize=30)


  ax = ax_list[1][1]
  im = ax.imshow( HI_dens_proj_ch, interpolation='bilinear', vmin=HI_min, vmax=HI_max )
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="5%", pad=0.05)
  fig.colorbar( im, cax=cax )
  ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
  #
  ax = ax_list[1][2]
  im = ax.imshow( HII_dens_proj_ch, interpolation='bilinear', vmin=HII_min, vmax=HII_max)
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="5%", pad=0.05)
  fig.colorbar( im, cax=cax )
  ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)

  ax = ax_list[1][3]
  im = ax.imshow( HeI_dens_proj_ch, interpolation='bilinear', vmin=HeI_min, vmax=HeI_max)
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="5%", pad=0.05)
  fig.colorbar( im, cax=cax )
  ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)

  ax = ax_list[1][4]
  im = ax.imshow( HeII_dens_proj_ch, interpolation='bilinear', vmin=HeII_min, vmax=HeII_max)
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="5%", pad=0.05)
  fig.colorbar( im, cax=cax )
  ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)

  ax = ax_list[1][5]
  im = ax.imshow( HeIII_dens_proj_ch, interpolation='bilinear', vmin=HeIII_min, vmax=HeIII_max)
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="5%", pad=0.05)
  fig.colorbar( im, cax=cax )
  ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)

  ax = ax_list[1][6]
  im = ax.imshow( e_dens_proj_ch, interpolation='bilinear', vmin=e_min, vmax=e_max )
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="5%", pad=0.05)
  fig.colorbar( im, cax=cax )
  ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)

  ax = ax_list[1][7]
  im = ax.imshow( gasEnergy_proj_ch, interpolation='bilinear', cmap='jet', vmin=ge_min, vmax=ge_max )
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="5%", pad=0.05)
  fig.colorbar( im, cax=cax )
  ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)

  if metals:
    ax = ax_list[1][8]
    im = ax.imshow( metal_dens_proj_ch, interpolation='bilinear', vmin=metal_min, vmax=metal_max )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar( im, cax=cax )
    ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
  #
  #
  # ax = ax_list[2][0]
  # im = ax.imshow( dens_proj_en_0, interpolation='bilinear', )
  # divider = make_axes_locatable(ax)
  # cax = divider.append_axes("right", size="5%", pad=0.05)
  # fig.colorbar( im, cax=cax )
  # ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
  # ax.set_ylabel( "No Cooling", fontsize=30)
  # ax.set_title( "ENZO Overdensity".format(current_z_ch), fontsize=30)
  #
  #
  # ax = ax_list[2][1]
  # im = ax.imshow( dens_proj_ch_0, interpolation='bilinear',)
  # divider = make_axes_locatable(ax)
  # cax = divider.append_axes("right", size="5%", pad=0.05)
  # fig.colorbar( im, cax=cax )
  # ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
  # ax.set_title( "CHOLLA Overdensity".format(current_z_ch), fontsize=30)
  #
  #
  # ax = ax_list[2][2]
  # im = ax.imshow( gasEnergy_proj_en_0, interpolation='bilinear', cmap='jet', vmin=ge_min, vmax=ge_max )
  # divider = make_axes_locatable(ax)
  # cax = divider.append_axes("right", size="5%", pad=0.05)
  # fig.colorbar( im, cax=cax )
  # ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
  # ax.set_title( "ENZO Temperature".format(current_z_ch), fontsize=30)
  #
  # ax = ax_list[2][3]
  # im = ax.imshow( gasEnergy_proj_ch_0, interpolation='bilinear', cmap='jet', vmin=ge_min, vmax=ge_max )
  # divider = make_axes_locatable(ax)
  # cax = divider.append_axes("right", size="5%", pad=0.05)
  # fig.colorbar( im, cax=cax )
  # ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
  # ax.set_title( "CHOLLA Temperature".format(current_z_ch), fontsize=30)

  #
  fig.tight_layout()
  fileName = 'chemistry_{0}.png'.format(nSnap)
  fig.savefig( outDir + fileName )
  print 'Saved image: ', fileName
  print ''
