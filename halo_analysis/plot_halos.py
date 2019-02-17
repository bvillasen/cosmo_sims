import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection





def plot_halos_positions( box_size, halo_catalog, dens, outDir, snap ):
  nHalos = halo_catalog.shape[0]
  plot_halos = False
  if nHalos > 0: plot_halos = True
  if plot_halos:
    radius, pos_x, pos_y, pos_z = halo_catalog[ :, 3:-1 ].T
    r_vir = halo_catalog[:,-1]
    pos_y = np.abs( pos_y - box_size )

  dens_proj = dens.sum( axis=0 )


  fig, axList = plt.subplots( 1, 2, sharex=True, sharey=True )
  fig.set_size_inches(20,10)
  plt.tight_layout()
  plt.subplots_adjust( hspace=0, wspace=0 )
  for i in [0,1]:
    if i == 0:
      ax = axList[i]
      img = np.log10( dens_proj  )
      ax.imshow(img, extent=[0,box_size,0,box_size], interpolation='bilinear', cmap='viridis'  )
      fig.axes[0].get_yaxis().set_visible(False)
      fig.axes[0].get_xaxis().set_visible(False)

    if i == 1:
      ax = axList[i]
      img = np.log10(dens_proj)
      ax.imshow(img, extent=[0,box_size,0,box_size], interpolation='bilinear', cmap='viridis'  )
      if plot_halos:
        circles = [ Circle( (pos_x[i], pos_y[i]), radius[i],) for i in range(nHalos) ]
        p = PatchCollection(circles, facecolors='none', edgecolors='w', linewidth=1, alpha=0.5 )
        ax.add_collection(p)

      fig.axes[1].get_yaxis().set_visible(False)
      fig.axes[1].get_xaxis().set_visible(False)
    # plt.axis('off')
  outFileName = outDir + 'densProj_{0}.png'.format(snap)
  fig.savefig( outFileName, pad_inches=0,  bbox_inches='tight', dpi=80 )


# from mpl_toolkits.axes_grid1 import make_axes_locatable
# import h5py as h5
#
# cosmo_dir = '/home/bruno/Desktop/Dropbox/Developer/cosmo_sims/'
# toolsDirectory = cosmo_dir + "tools/"
# sys.path.append( toolsDirectory )
# from halo_tools import load_asciiFiles
#
#
# dataDir = '/home/bruno/Desktop/data/'
# # # inDir = dataDir +  'cosmo_sims/256_gadget/'
# inDir = dataDir +  'cosmo_sims/cholla_pm/cosmo_256_dm/'
# outDir = inDir
#
# snap = 19
#
# inFileName = 'data/particles_{0}.h5'.format(snap)
# inFile = h5.File( inDir + inFileName )
#
# dens = inFile['density'][...]
#
# inFile.close()
#
# # print ' Loading halo catalogs'
# halos_Dir = inDir + "halos/"
# snapshots = [ 19 ]
# h_data_all = load_asciiFiles( snapshots, 8, halos_Dir )
#
#
# h = 0.687
# box_size = 115.
#
#
#
# dens_proj = dens.sum( axis=0 )
# #   # if show_halos:
# h_data = h_data_all[snap]
# nHalos = h_data['nHalos']
# h_mass = h_data['mbound_vir']
# h_pos_x = h_data['x']
# h_pos_y = h_data['y']
# h_pos_z = h_data['z']
# h_radius = h_data['rvir'] * 1e-3
# h_pos_y = np.abs( h_pos_y - box_size )
#
# # Filter halos
# mass_min = 1e12
# flt_mass = h_mass > mass_min
# h_filter = flt_mass
# h_mass = h_mass[h_filter]
# h_pos_x = h_pos_x[h_filter]
# h_pos_y = h_pos_y[h_filter]
# h_pos_z = h_pos_z[h_filter]
# h_radius = h_radius[h_filter]
# nHalos = len(h_mass)
#
#
#
#
#
# fig, axList = plt.subplots( 1, 2, sharex=True, sharey=True )
# fig.set_size_inches(20,10)
# plt.tight_layout()
# plt.subplots_adjust( hspace=0, wspace=0 )
# for i in [0,1]:
#   if i == 0:
#     ax = axList[i]
#     img = np.log10( dens_proj  )
#     ax.imshow(img, extent=[0,box_size,0,box_size], interpolation='bilinear', cmap='viridis'  )
#     fig.axes[0].get_yaxis().set_visible(False)
#     fig.axes[0].get_xaxis().set_visible(False)
#
#   if i == 1:
#     ax = axList[i]
#     img = np.log10(dens_proj)
#     ax.imshow(img, extent=[0,box_size,0,box_size], interpolation='bilinear', cmap='viridis'  )
#     # if show_halos:
#     #   # colors = np.log10(h_mass)
#     circles = [ Circle( (h_pos_x[i], h_pos_y[i]), h_radius[i],) for i in range(nHalos) ]
#     p = PatchCollection(circles, facecolors='none', edgecolors='w', linewidth=1, alpha=0.5 )
#     #   # # p.set_array(np.array(colors))
#     ax.add_collection(p)
#     # # plt.colorbar()
#     # ax.set_axis_off()
#     # if show_halos:
#     #   # colors = np.log10(h_mass)
#     #   circles = [ Circle( (h_pos_x[i], h_pos_y[i]), h_radius[i],) for i in range(nHalos) ]
#     #   p = PatchCollection(circles, facecolors='none', edgecolors='w', linewidth=1, alpha=0.5 )
#     #   # # p.set_array(np.array(colors))
#     #   ax.add_collection(p)
#     # # plt.colorbar()
#     # ax.set_axis_off()
#     # text = r'z = {0:.1f}'.format(current_z_1)
#     # if show_halos: text = 'z = {0:.1f} \nn_halos: {1}'.format(current_z, nHalos)
#     # ax.text(box_size*0.05, box_size*0.90, text, fontsize=20, color='white',
#     #         bbox={'facecolor':'black', 'edgecolor':'white', 'alpha':0.5, })
#
#     fig.axes[1].get_yaxis().set_visible(False)
#     fig.axes[1].get_xaxis().set_visible(False)
#   # plt.axis('off')
# outFileName = outDir + 'densProj_{0}.png'.format(snap)
# fig.savefig( outFileName, pad_inches=0,  bbox_inches='tight', dpi=80 )
