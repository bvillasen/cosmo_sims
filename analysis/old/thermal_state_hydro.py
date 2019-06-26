import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5

# dataDir = '/raid/bruno/data/cosmo_sims/'
dataDir = '/home/bruno/Desktop/data/'
cosmo_dir = '/home/bruno/Desktop/Dropbox/Developer/cosmo_sims/'
outDir = cosmo_dir + 'figures/phase_diagram/'
toolsDirectory = cosmo_dir + "tools/"
sys.path.extend([toolsDirectory ] )
from load_data_cholla import load_snapshot_data
from load_data_enzo import load_snapshot_enzo
from internal_energy import get_internal_energy, get_temp
from phase_diagram import get_phase_diagram

dataDir = '/raid/bruno/data/'



enzoDir = dataDir + 'cosmo_sims/enzo/256_hydro/h5_files/'
chollaDir = dataDir + 'cosmo_sims/cholla_pm/256_hydro/data_gravCPU/'

outDir = cosmo_dir + 'figures/phase_diagram/hydro/'

gamma = 5./3
nPoints = 256
nx = nPoints
ny = nPoints
nz = nPoints

dv = (115000./nPoints)**3


nbins = 1000

nSnap = 0
snapshots = range(0,31)
for nSnap in snapshots:

  data_cholla = load_snapshot_data( str(nSnap), chollaDir )
  current_z_ch = data_cholla['current_z']
  dens = data_cholla['gas']['density'][...]
  GE = data_cholla['gas']['GasEnergy'][...] / dens
  dens_mean = dens.mean()
  rho_ch = dens.reshape( nx*ny*nz ) / dens_mean
  temp = get_temp(GE*1e6, gamma)
  temp_ch = temp.reshape( nx*ny*nz )
  temp_avrg_ch = temp_ch.mean()
  # z_list.append(current_z_ch)
  # temps.append( temp_avrg_ch)
  x_ch, y_ch, z_ch = get_phase_diagram( rho_ch, temp_ch , nbins ) 

  data_enzo = load_snapshot_enzo( nSnap, enzoDir )
  current_z_enzo = data_enzo['current_z']
  dens = data_enzo['gas']['density'][...]
  GE = data_enzo['gas']['GasEnergy'][...] / dens
  dens_mean = dens.mean()
  rho_enzo = dens.reshape( nx*ny*nz ) / dens_mean
  temp = get_temp(GE*1e6, gamma)
  # temp_factor = 1.23855450891
  # temp_enzo *= temp_factor
  temp_avrg_enzo = temp.mean()
  temp_enzo = temp.reshape( nx*ny*nz )
  # temps_enzo.append( temp_avrg_enzo)
  x_en, y_en, z_en = get_phase_diagram( rho_enzo, temp_enzo , nbins ) 


  fig, ax_l = plt.subplots(nrows=1, ncols=2, figsize=(16,8))

  ax = ax_l[0]
  c = ax.scatter( y_en, x_en, c = np.log10(z_en), s=1, )
  ax.set_ylabel(r'Log Temperature  $[K]$', fontsize=15 )
  ax.set_xlabel(r'Log Gas Overdensity', fontsize=15 )
  ax.set_title( "Z = {0:.2f}           ENZO ".format( current_z_ch) )
  ax.set_xlim(-2, 3)
  ax.set_ylim(-1, 8)
  # 
  ax = ax_l[1]
  c = ax.scatter( y_ch, x_ch, c = np.log10(z_ch), s=1,  )
  ax.set_ylabel(r'Log Temperature  $[K]$', fontsize=15 )
  ax.set_xlabel(r'Log Gas Overdensity', fontsize=15 )
  ax.set_title( "CHOLLA        DE_t=0.001  ")
  ax.set_xlim(-2, 3)
  ax.set_ylim(-1, 8)

  fig.tight_layout()
  fileName = 'thermal_state_hydro_{0}.png'.format(nSnap)
  fig.savefig( outDir + fileName )
  print 'Saved image: ', fileName
  print ''







# z_list, temps_enzo, temps, temps_0, temps_1, temps_2, temps_3, temps_4 = [], [], [], [], [], [], [], []
# for nSnap in snapshots:
#   snapKey = str( nSnap )
#
#   data_cholla = load_snapshot_data( str(nSnap), chollaDir )
#   current_z_ch = data_cholla['current_z']
#   current_a_ch = data_cholla['current_a']
#   dens_gas_cholla = data_cholla['gas']['density'][...]
#   GasEnergy = data_cholla['gas']['GasEnergy'][...] / dens_gas_cholla
#   dens_mean = dens_gas_cholla.mean()
#   rho_ch = dens_gas_cholla.reshape( nx*ny*nz ) / dens_mean
#   temp_ch = get_temp(GasEnergy*1e6, gamma)
#   temp_avrg_ch = (temp_ch*dens_gas_cholla/dens_gas_cholla.mean()).mean()
#   # temp_avrg_ch = temp_ch.mean()
#   temp_ch = temp_ch.reshape( nx*ny*nz )
#   z_list.append(current_z_ch)
#   temps.append( temp_avrg_ch)
#
#   data_cholla = load_snapshot_data( str(nSnap), chollaDir_0 )
#   current_z_ch_0 = data_cholla['current_z']
#   current_a_ch = data_cholla['current_a']
#   dens_gas_cholla = data_cholla['gas']['density'][...]
#   GasEnergy = data_cholla['gas']['GasEnergy'][...] / dens_gas_cholla
#   dens_mean = dens_gas_cholla.mean()
#   rho_ch_0 = dens_gas_cholla.reshape( nx*ny*nz ) / dens_mean
#   temp_ch_0 = get_temp(GasEnergy*1e6, gamma)
#   temp_avrg_ch_0 = (temp_ch_0*dens_gas_cholla/dens_gas_cholla.mean()).mean()
#   # temp_avrg_ch_0 = temp_ch_0.mean()
#   temp_ch_0 = temp_ch_0.reshape( nx*ny*nz )
#   temps_0.append( temp_avrg_ch_0)
#
#   # data_cholla = load_snapshot_data( str(nSnap), chollaDir_1 )
#   # current_z_ch_1 = data_cholla['current_z']
#   # current_a_ch = data_cholla['current_a']
#   # dens_gas_cholla = data_cholla['gas']['density'][...]
#   # GasEnergy = data_cholla['gas']['GasEnergy'][...] / dens_gas_cholla
#   # dens_mean = dens_gas_cholla.mean()
#   # rho_ch_1 = dens_gas_cholla.reshape( nx*ny*nz ) / dens_mean
#   # temp_ch_1 = get_temp(GasEnergy*1e6, gamma)
#   # temp_avrg_ch_1 = (temp_ch_1*dens_gas_cholla/dens_gas_cholla.mean()).mean()
#   # # temp_avrg_ch_1 = temp_ch_1.mean()
#   # temp_ch_1 = temp_ch_1.reshape( nx*ny*nz )
#   # temps_1.append( temp_avrg_ch_1)
# #
#   data_cholla = load_snapshot_data( str(nSnap), chollaDir_2 )
#   current_z_ch_2 = data_cholla['current_z']
#   current_a_ch = data_cholla['current_a']
#   dens_gas_cholla = data_cholla['gas']['density'][...]
#   GasEnergy = data_cholla['gas']['GasEnergy'][...] / dens_gas_cholla
#   dens_mean = dens_gas_cholla.mean()
#   rho_ch_2 = dens_gas_cholla.reshape( nx*ny*nz ) / dens_mean
#   temp_ch_2 = get_temp(GasEnergy*1e6, gamma)
#   temp_avrg_ch_2 = (temp_ch_2*dens_gas_cholla/dens_gas_cholla.mean()).mean()
#   # temp_avrg_ch_2 = temp_ch_2.mean()
#   temp_ch_2 = temp_ch_2.reshape( nx*ny*nz )
#   temps_2.append( temp_avrg_ch_2)
# #
#   data_cholla = load_snapshot_data( str(nSnap), chollaDir_3 )
#   current_z_ch_3 = data_cholla['current_z']
#   current_a_ch = data_cholla['current_a']
#   dens_gas_cholla = data_cholla['gas']['density'][...]
#   GasEnergy = data_cholla['gas']['GasEnergy'][...] / dens_gas_cholla
#   dens_mean = dens_gas_cholla.mean()
#   rho_ch_3 = dens_gas_cholla.reshape( nx*ny*nz ) / dens_mean
#   temp_ch_3 = get_temp(GasEnergy*1e6, gamma)
#   temp_avrg_ch_3 = (temp_ch_3*dens_gas_cholla/dens_gas_cholla.mean()).mean()
#   # temp_avrg_ch_3 = temp_ch_3.mean()
#   temp_ch_3 = temp_ch_3.reshape( nx*ny*nz )
#   temps_3.append( temp_avrg_ch_3)
#
#   data_cholla = load_snapshot_data( str(nSnap), chollaDir_4 )
#   current_z_ch_4 = data_cholla['current_z']
#   current_a_ch = data_cholla['current_a']
#   dens_gas_cholla = data_cholla['gas']['density'][...]
#   GasEnergy = data_cholla['gas']['GasEnergy'][...] / dens_gas_cholla
#   dens_mean = dens_gas_cholla.mean()
#   rho_ch_4 = dens_gas_cholla.reshape( nx*ny*nz ) / dens_mean
#   temp_ch_4 = get_temp(GasEnergy*1e6, gamma)
#   temp_avrg_ch_4 = (temp_ch_4*dens_gas_cholla/dens_gas_cholla.mean()).mean()
#   # temp_avrg_ch_4 = temp_ch_4.mean()
#   temp_ch_4 = temp_ch_4.reshape( nx*ny*nz )
#   temps_4.append( temp_avrg_ch_4)
#
#   data_enzo = load_snapshot_enzo( nSnap, enzoDir )
#   current_z_enzo = data_enzo['current_z']
#   current_a_enzo = data_enzo['current_a']
#   dens_gas_enzo = data_enzo['gas']['density'][...]
#   GasEnergy = data_enzo['gas']['GasEnergy'][...] / dens_gas_enzo
#   dens_mean = dens_gas_enzo.mean()
#   rho_enzo = dens_gas_enzo.reshape( nx*ny*nz ) / dens_mean
#   temp_enzo = get_temp(GasEnergy*1e6, gamma)
#   # if nSnap == 0: temp_factor = 172/temp_enzo.mean()
#   temp_factor = 1.23855450891
#   temp_enzo *= temp_factor
#   temp_avrg_enzo = (temp_enzo*dens_gas_enzo/dens_gas_enzo.mean()).mean()
#   # temp_avrg_enzo = temp_enzo.mean()
#   temp_enzo = temp_enzo.reshape( nx*ny*nz )
#   temps_enzo.append( temp_avrg_enzo)
#
#
#   nbins = 2000
#   x_enzo, y_enzo, z_enzo = get_phase_diagram( rho_enzo, temp_enzo, nbins )
#   x_ch, y_ch, z_ch = get_phase_diagram( rho_ch, temp_ch , nbins )
#   x_ch_0, y_ch_0, z_ch_0 = get_phase_diagram( rho_ch_0, temp_ch_0 , nbins )
#   # x_ch_1, y_ch_1, z_ch_1 = get_phase_diagram( rho_ch_1, temp_ch_1 , nbins )
#   x_ch_2, y_ch_2, z_ch_2 = get_phase_diagram( rho_ch_2, temp_ch_2 , nbins )
#   x_ch_3, y_ch_3, z_ch_3 = get_phase_diagram( rho_ch_3, temp_ch_3 , nbins )
#   x_ch_4, y_ch_4, z_ch_4 = get_phase_diagram( rho_ch_4, temp_ch_4 , nbins )
#
#
#   fig, ax_l = plt.subplots(nrows=2, ncols=4, figsize=(32,16))
#
#   ax = ax_l[0][0]
#   c = ax.scatter( y_enzo, x_enzo, c = np.log10(z_enzo), s=1,  )
#   # plt.colorbar(c)
#   ax.set_ylabel(r'Log Temperature $[K]$', fontsize=15 )
#   ax.set_xlabel(r'Log Gas Overdensity', fontsize=15 )
#   # ax1.set_xlabel(r'Log Gas Density $[h^{2}{\rm M_{\odot}} kpc^{-3}]$', fontsize=15 )
#   ax.set_title( "Enzo   Z = {0:.2f}   T_mean={1:.2f}".format(current_z_enzo, temp_avrg_enzo))
#   ax.set_xlim(-2, 3)
#   ax.set_ylim(-1, 8)
# #   #
#   ax = ax_l[0][1]
#   c = ax.scatter( y_ch, x_ch, c = np.log10(z_ch), s=1,  )
#   # plt.colorbar(c)
#   ax.set_ylabel(r'Log Temperature $[K]$', fontsize=15 )
#   ax.set_xlabel(r'Log Gas Overdensity', fontsize=15 )
#   # ax2.set_xlabel(r'Log Gas Density $[h^{2}{\rm M_{\odot}} kpc^{-3}]$', fontsize=15 )
#   ax.set_title( "CHOLLA GRAV_1  Z = {0:.2f} DE_t=0.001 delta_a=0.001 T_mean={1:.2f}".format(current_z_ch, temp_avrg_ch))
#   ax.set_xlim(-2, 3)
#   ax.set_ylim(-1, 8)
# #
#   ax = ax_l[0][2]
#   c = ax.scatter( y_ch_0, x_ch_0, c = np.log10(z_ch_0), s=1,  )
#   # plt.colorbar(c)
#   ax.set_ylabel(r'Log Temperature $[K]$', fontsize=15 )
#   ax.set_xlabel(r'Log Gas Overdensity', fontsize=15 )
#   # ax2.set_xlabel(r'Log Gas Density $[h^{2}{\rm M_{\odot}} kpc^{-3}]$', fontsize=15 )
#   ax.set_title( "CHOLLA DE_SYNC_FIRST  Z = {0:.2f} DE_t=0.001 delta_a=0.001 T_mean={1:.2f}".format(current_z_ch_0, temp_avrg_ch_0))
#   ax.set_xlim(-2, 3)
#   ax.set_ylim(-1, 8)
#
# #
# # #
#   # ax = ax_l[0][3]
#   # c = ax.scatter( y_ch_1, x_ch_1, c = np.log10(z_ch_1), s=1,  )
#   # # plt.colorbar(c)
#   # ax.set_ylabel(r'Log Temperature $[K]$', fontsize=15 )
#   # ax.set_xlabel(r'Log Gas Overdensity', fontsize=15 )
#   # # ax2.set_xlabel(r'Log Gas Density $[h^{2}{\rm M_{\odot}} kpc^{-3}]$', fontsize=15 )
#   # ax.set_title( "CHOLLA DE_SYNC_FIRST NO_GE_TERM  Z = {0:.2f} DE_t=0.001 delta_a=0.001 T_mean={1:.2f}".format(current_z_ch_1, temp_avrg_ch_1))
#   # ax.set_xlim(-2, 3)
#   # ax.set_ylim(-1, 8)
# # #
# # #
#   ax = ax_l[1][0]
#   c = ax.scatter( y_ch_2, x_ch_2, c = np.log10(z_ch_2), s=1,  )
#   # plt.colorbar(c)
#   ax.set_ylabel(r'Log Temperature $[K]$', fontsize=15 )
#   ax.set_xlabel(r'Log Gas Overdensity', fontsize=15 )
#   # ax2.set_xlabel(r'Log Gas Density $[h^{2}{\rm M_{\odot}} kpc^{-3}]$', fontsize=15 )
#   ax.set_title( "CHOLLA Ek>0.4  Z = {0:.2f} DE_t=0.001 delta_a=0.001 T_mean={1:.2f}".format(current_z_ch_2, temp_avrg_ch_2))
#   ax.set_xlim(-2, 3)
#   ax.set_ylim(-1, 8)
# #
#   ax = ax_l[1][1]
#   c = ax.scatter( y_ch_3, x_ch_3, c = np.log10(z_ch_3), s=1,  )
#   # plt.colorbar(c)
#   ax.set_ylabel(r'Log Temperature $[K]$', fontsize=15 )
#   ax.set_xlabel(r'Log Gas Overdensity', fontsize=15 )
#   # ax2.set_xlabel(r'Log Gas Density $[h^{2}{\rm M_{\odot}} kpc^{-3}]$', fontsize=15 )
#   ax.set_title( "CHOLLA Ek>0.4  Z = {0:.2f} DE_t=0.002 delta_a=0.001 T_mean={1:.2f}".format(current_z_ch_3, temp_avrg_ch_3))
#   ax.set_xlim(-2, 3)
#   ax.set_ylim(-1, 8)
# #
#   ax = ax_l[1][2]
#   c = ax.scatter( y_ch_4, x_ch_4, c = np.log10(z_ch_4), s=1,  )
#   # plt.colorbar(c)
#   ax.set_ylabel(r'Log Temperature $[K]$', fontsize=15 )
#   ax.set_xlabel(r'Log Gas Overdensity', fontsize=15 )
#   # ax2.set_xlabel(r'Log Gas Density $[h^{2}{\rm M_{\odot}} kpc^{-3}]$', fontsize=15 )
#   ax.set_title( "CHOLLA  Ek>0.4  Z = {0:.2f} DE_t=0.005 delta_a=0.001 T_mean={1:.2f}".format(current_z_ch_4, temp_avrg_ch_4))
#   ax.set_xlim(-2, 3)
#   ax.set_ylim(-1, 8)
#
# #
# #
# #
#   #
#   fig.tight_layout()
#   fileName = 'thermal_state_enzo_{0}.png'.format(nSnap)
#   fig.savefig( outDir + fileName )
#   print 'Saved image: ', fileName
#   print ''

# plt.figure(0)
# plt.clf()
# plt.plot( z_list, temps,  c='b', label='Cholla DE_t=0.001  da=0.001' )
# plt.plot( z_list, temps_0, '--', c='b', label='Cholla DE_t=0.001  da=0.0001' )
# plt.plot( z_list, temps_1,  c='r', label='Cholla DE_t=0.01  da=0.0001' )
# plt.plot( z_list, temps_2,  c='m' ,label='Cholla DE_t=0.05  da=0.001' )
# plt.plot( z_list, temps_3,  '--', c='m' ,label='Cholla DE_t=0.05  da=0.0001' )
# plt.plot( z_list, temps_4,  '--', c='y' ,label='Cholla DE_t=0.05  da=0.00002' )
# plt.plot( z_list, temps_enzo, linewidth=2, c='g', label='Enzo' )
# plt.xlabel( 'Z' )
# plt.title('Volume Wheighted')
# plt.title('Mass Wheighted')
# # plt.ylabel( r'Mean Temperature  [$K$]')
# plt.legend(loc=1)
# plt.yscale('log')
# # plt.xscale('log')
# plt.ylim(1, 1e7)
# # plt.savefig(outDir + 'temperature_DE_v_1.png')
# plt.savefig(outDir + 'temperature_DE_m_1.png')
#
