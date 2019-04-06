import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
from power_spectrum import get_power_spectrum

dataDir = '/raid/bruno/data/'
# dataDir = '/home/bruno/Desktop/data/'
cosmo_dir = '/home/bruno/Desktop/Dropbox/Developer/cosmo_sims/'
toolsDirectory = cosmo_dir + "tools/"
sys.path.extend([toolsDirectory ] )
from load_data_cholla import load_snapshot_data
from load_data_enzo import load_snapshot_enzo

enzoDir = dataDir + 'cosmo_sims/enzo/256_hydro/h5_files/'
chollaDir = dataDir + 'cosmo_sims/cholla_pm/256_hydro/data/'
outDir = cosmo_dir + 'figures/power_hydro/'

# set simulation volume dimentions
nPoints = 256
nz, ny, nx = nPoints, nPoints, nPoints
nCells  = nx*ny*nz
Lbox = 115.0   #Mpc/h
h = 0.6774
Lx = Lbox
Ly = Lbox
Lz = Lbox
dx, dy, dz = Lx/(nx), Ly/(ny), Lz/(nz )

n_kSamples = 12


fileName = outDir + 'power_hydro_256_0.png'



n_snapshots = 31
snapshots = range( 0, n_snapshots, 2 )
fig = plt.figure(0)
fig.set_size_inches(20,10)
fig.clf()


gs = plt.GridSpec(5, 2)
gs.update(hspace=0.05, wspace=0.08, )
ax1 = plt.subplot(gs[0:4, 0])
ax2 = plt.subplot(gs[4:5, 0])

ax3 = plt.subplot(gs[0:4, 1])
ax4 = plt.subplot(gs[4:5, 1])
ax4.axhline( y=0., color='r', linestyle='--',  )
ax2.axhline( y=0., color='r', linestyle='--',  )

colors = ['b', 'y', 'g', 'c', 'm', 'b', 'y', 'g', 'c', 'm','b', 'y', 'g', 'c', 'm', 'b', 'y', 'g', 'c', 'm' ]

for i,nSnap in enumerate(snapshots):
  print " Cholla: ", nSnap
  snapKey = str( nSnap )
  # if i not in [9]: continue
  data_cholla = load_snapshot_data( snapKey, chollaDir )
  current_z_ch = data_cholla['current_z']
  dens_dm_cholla = data_cholla['dm']['density'][...]
  dens_gas_cholla = data_cholla['gas']['density'][...]
  print current_z_ch
  dens_ch = (dens_dm_cholla + dens_gas_cholla).mean()
  print dens_ch
  
  ps_dm_cholla, k_vals, count_dm_cholla = get_power_spectrum( dens_dm_cholla, Lbox, nx, ny, nz, dx, dy, dz,  n_kSamples=n_kSamples)
  ps_gas_cholla, k_vals, count_gas_cholla = get_power_spectrum( dens_gas_cholla, Lbox, nx, ny, nz, dx, dy, dz,  n_kSamples=n_kSamples)

  c= colors[i]
  label = 'z = {0:.1f}'.format(current_z_ch)
  ax1.plot( k_vals, ps_dm_cholla,  c=c, linewidth=2, label=label )
  ax3.plot( k_vals, ps_gas_cholla, c=c,  linewidth=2, label=label )

  print ' Enzo: ', nSnap
  data_enzo = load_snapshot_enzo( nSnap, enzoDir, dm=True)
  current_a_enzo = data_enzo['current_a']
  current_z_enzo = data_enzo['current_z']
  print current_z_enzo
  dens_gas_enzo = data_enzo['gas']['density'][...]
  dens_dm_enzo = data_enzo['dm']['density'][...]
  dens_en = (dens_dm_enzo + dens_gas_enzo).mean()
  print dens_en

  ps_dm_enzo, k_vals, count_dm_enzo = get_power_spectrum( dens_dm_enzo, Lbox, nx, ny, nz, dx, dy, dz,  n_kSamples=n_kSamples)
  ps_gas_enzo, k_vals, count_gas_enzo = get_power_spectrum( dens_gas_enzo, Lbox, nx, ny, nz, dx, dy, dz,  n_kSamples=n_kSamples)
  ax1.plot( k_vals, ps_dm_enzo, '--', c='k', linewidth=1 )
  ax3.plot( k_vals, ps_gas_enzo, '--', c='k', linewidth=1 )


  error_dm = (ps_dm_cholla - ps_dm_enzo) / ps_dm_cholla
  error_gas = (ps_gas_cholla - ps_gas_enzo) / ps_gas_cholla


  ax2.plot( k_vals, error_dm , c=c, alpha=0.9)
  ax4.plot( k_vals, error_gas , c=c, alpha=0.9)

ax2.set_ylim( -0.2, 0.2)
ax4.set_ylim( -0.2, 0.2)

ax1.set_ylabel( r'$P(k) $', fontsize=17)
ax2.set_ylabel( 'Error', fontsize=15)
ax2.set_xlabel( r'$k \, \, [h Mpc^{-1}]$', fontsize=17)
ax1.legend( loc=3)
ax4.set_xlabel( r'$k \, \, [h Mpc^{-1}]$', fontsize=17)
ax3.legend( loc=3)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax3.set_xscale('log')
ax3.set_yscale('log')
ax2.set_xscale('log')
ax4.set_xscale('log')
ax1.set_title('DM Power Spectrum',  fontsize=18)
ax3.set_title('Gas Power Spectrum', fontsize=18)
# ax1.xlim()
fig.savefig( fileName,  pad_inches=0.1,  bbox_inches='tight', dpi=80)
print 'Saved Image: ', fileName


# np.savetxt('outputs_nyx_hydro.txt', current_a_list)



#
# # #Cholla fie names
# partFileName = dataDir + 'cosmo_sims/cholla_pm/cosmo_256_hydro/data_particles_PLMP_01.h5'
# gridFileName = dataDir + 'cosmo_sims/cholla_pm/cosmo_256_hydro/data_grid_PLMP_01.h5'
#
#
# add_label = True
# snapshots_cholla = range(1,30, 2)
# # snapshots_cholla = [ 0, 1, 3, 5, 7, 9, 11,13, 15, 17, 19, 21, 23, 25, 27, 29]
# # snapshots_cholla = snapshots_nyx
# for nSnap in snapshots_cholla:
#   print " Cholla: ", nSnap
#   snapKey = str( nSnap )
#   data_cholla, nSnapshots = load_snapshot_data( snapKey, gridFileName, partFileName )
#   current_z_ch = data_cholla['dm']['current_z']
#   dens_dm_cholla = data_cholla['dm']['density'][...]
#   dens_gas_cholla = data_cholla['grid']['density'][...]
#
#   ps_dm_cholla_PLMP, k_vals = get_power_spectrum( dens_dm_cholla, Lbox, nx, ny, nz, dx, dy, dz,  n_kSamples=n_kSamples)
#   ps_gas_cholla_PLMP, k_vals = get_power_spectrum( dens_gas_cholla, Lbox, nx, ny, nz, dx, dy, dz,  n_kSamples=n_kSamples)
#   #
#   # label = 'z = {0:.1f}'.format(current_z_ch)
#   label = "PLMP"
#   if add_label:
#     ax1.plot( k_vals, ps_dm_cholla_PLMP,  c='r', linewidth=1, label=label )
#     ax2.plot( k_vals, ps_gas_cholla_PLMP, c='r',  linewidth=1, label=label )
#     add_label = False
#   else:
#     ax1.plot( k_vals, ps_dm_cholla_PLMP,  c='r', linewidth=1 )
#     ax2.plot( k_vals, ps_gas_cholla_PLMP, c='r',  linewidth=1 )
#
#
# # #Cholla fie names
# partFileName = dataDir + 'cosmo_sims/cholla_pm/cosmo_256_hydro/data_particles_PLMC_01.h5'
# gridFileName = dataDir + 'cosmo_sims/cholla_pm/cosmo_256_hydro/data_grid_PLMC_01.h5'
#
#
# add_label = True
# snapshots_cholla = range(1,30, 2)
# # snapshots_cholla = [ 0, 1, 3, 5, 7, 9, 11,13, 15, 17, 19, 21, 23, 25, 27, 29]
# # snapshots_cholla = snapshots_nyx
# for nSnap in snapshots_cholla:
#   print " Cholla: ", nSnap
#   snapKey = str( nSnap )
#   data_cholla, nSnapshots = load_snapshot_data( snapKey, gridFileName, partFileName )
#   current_z_ch = data_cholla['dm']['current_z']
#   dens_dm_cholla = data_cholla['dm']['density'][...]
#   dens_gas_cholla = data_cholla['grid']['density'][...]
#
#   ps_dm_cholla_PLMC, k_vals = get_power_spectrum( dens_dm_cholla, Lbox, nx, ny, nz, dx, dy, dz,  n_kSamples=n_kSamples)
#   ps_gas_cholla_PLMC, k_vals = get_power_spectrum( dens_gas_cholla, Lbox, nx, ny, nz, dx, dy, dz,  n_kSamples=n_kSamples)
#   #
#   # label = 'z = {0:.1f}'.format(current_z_ch)
#   label = "PLMC"
#   if add_label:
#     ax1.plot( k_vals, ps_dm_cholla_PLMC,  c='b', linewidth=1, label=label )
#     ax2.plot( k_vals, ps_gas_cholla_PLMC, c='b',  linewidth=1, label=label )
#     add_label = False
#   else:
#     ax1.plot( k_vals, ps_dm_cholla_PLMC,  c='b', linewidth=1 )
#     ax2.plot( k_vals, ps_gas_cholla_PLMC, c='b',  linewidth=1 )
#
#
#
# # #Cholla fie names
# partFileName = dataDir + 'cosmo_sims/cholla_pm/cosmo_256_hydro/data_particles_PPMP_01.h5'
# gridFileName = dataDir + 'cosmo_sims/cholla_pm/cosmo_256_hydro/data_grid_PPMP_01.h5'
# snapshots_cholla = range(1,10, 2)
# # snapshots_cholla = [ 0, 1, 3, 5, 7, 9, 11,13, 15, 17, 19, 21, 23, 25, 27, 29]
# # snapshots_cholla = snapshots_nyx
# add_label = True
# for nSnap in snapshots_cholla:
#   print " Cholla: ", nSnap
#   snapKey = str( nSnap )
#   data_cholla, nSnapshots = load_snapshot_data( snapKey, gridFileName, partFileName )
#   current_z_ch = data_cholla['dm']['current_z']
#   dens_dm_cholla = data_cholla['dm']['density'][...]
#   dens_gas_cholla = data_cholla['grid']['density'][...]
#
#   ps_dm_cholla_PPMP, k_vals = get_power_spectrum( dens_dm_cholla, Lbox, nx, ny, nz, dx, dy, dz,  n_kSamples=n_kSamples)
#   ps_gas_cholla_PPMP, k_vals = get_power_spectrum( dens_gas_cholla, Lbox, nx, ny, nz, dx, dy, dz,  n_kSamples=n_kSamples)
#   #
#   # label = 'z = {0:.1f}'.format(current_z_ch)
#   label = "PPMP"
#   if add_label:
#     ax1.plot( k_vals, ps_dm_cholla_PPMP,  c='g', linewidth=1, label=label )
#     ax2.plot( k_vals, ps_gas_cholla_PPMP, c='g',  linewidth=1, label=label )
#     add_label = False
#   else:
#     ax1.plot( k_vals, ps_dm_cholla_PPMP,  c='g', linewidth=1 )
#     ax2.plot( k_vals, ps_gas_cholla_PPMP, c='g',  linewidth=1 )






# # plt.show()0
#
#
# inFile.close()
#



# A = 4*np.pi*k_mag**2
# err = ( int_val - A )/A







# int_val = 0
# for theta in theta_vals:
#   for phi in phi_vals:
#     x = r*np.sin( theta )*np.cos(phi)
#     y = r*np.sin( theta )*np.sin(phi)
#     z = r*np.cos( theta )
#     k_vec = np.array([z, y, x])
#     r2 = get_interp_val( k_vec )
#     int_val += r2*np.cos(theta)*d_tetha*d_phi
# A = 4*np.pi*r**2
# err = ( int_val - A )/A

# fig = plt.figure(0)
# plt.plot( fft_kx )
# plt.show()

# plt.imshow(gauss[nPoints/2,:,:])
# plt.show()
#
# fig = plt.figure(1)
# plt.imshow(power[:,:,nPoints/2])
# plt.show()






# bins = 1.
# tbins = 400.
#
# density = np.random.randn(400,400,400)
#
# x, y, z = np.mgrid[0:400, 0:400, 0:400]
# x = x - 199.5
# y = y - 199.5
# z = z - 199.5
#
# dist = np.sqrt(x**2+y**2+z**2)
#
# FT = np.fft.fftn(density)
# power = FT.real*FT.real + FT.imag*FT.imag
#
# P = power.reshape(np.size(power))
# dist = dist.reshape(np.size(dist))
#
# intervals = np.array([nn*bins for nn in range(0,int(tbins)+1)])
#
# p = np.histogram(dist, bins=intervals, weights=P)[0]
# pd = np.histogram(dist, bins=intervals)[0]
# pd.astype('float')
# p = p/pd
#
# plt.figure()
# plt.plot(2.*np.pi/intervals[1:], p)
# plt.show()
# a_list = []
# for nSnap in range(nSnapshots):
#   snapKey = str( nSnap )
#   current_a = inFile[snapKey]['current_a'][0]
#   a_list.append(current_a)
#
# a_list = np.array(a_list)
# np.savetxt('outputs_alin_50.txt', a_list)

# nSnap = 0
# snapKey = str( nSnap )
# pos_x = inFile[snapKey]['pos_x'][...]
# dens = inFile[snapKey]['density'][...]
# pot  = inFile[snapKey]['potential'][...]
# # time = inFile[snapKey]['t'][0]
#
# # define a vector of cosmological parameters:
# my_cosmo = {'flat': True, 'H0': 70.0, 'Om0': 0.27, 'Ob0': 0.045714, 'sigma8': 0.82, 'ns': 0.96}
#
# # set my_cosmo to be the current cosmology
# cosmo = cosmology.setCosmology('planck15')
# cosmo.interpolation = False
# dens_mean = cosmo.rho_m(0)

# snapshots, inFile = load_file( inDir + inFileName)
# nSnapshots = len(snapshots)
#
# nSnap = 0
# snapKey = str(nSnap)
# dens = inFile[snapKey]['density'][...]
# pos_x = inFile[snapKey]['pos_x'][...]
# inFile.close()
#
# for nSnap in range(nSnapshots):
#   print ' Plotting snap: {0}'.format( nSnap )
#   snapKey = str(nSnap)
#   # dens = inFile[snapKey]['grav_density'][...]
#   if particles:dens = inFile[snapKey]['density'][...]
#   pot  = inFile[snapKey]['potential'][...]
#   time = inFile[snapKey]['t'][0]
#   # print time
#   dims = dens.shape
#   nz, ny, nx = dims
#
#   if time == 0: r_interp = 0.2
#   else:  r_interp = np.interp(time, t_all, radius_all )
#   # print time, r_interp
#   circle1 = plt.Circle((0.5, 0.5), r_interp, color='k', fill=False)
#
#   fig  = plt.figure(0)
#   fig.clf()
#   ax = plt.gca()
#   cut = int(0.5*ny)
#   # data = dens[:,cut,:]
#   data = np.sum(dens, axis=1)
#   data = np.log10(data + 1)
#   img = ax.imshow( data, extent=[0,1,0,1] )
#   fig.colorbar( img )
#   ax.set_title('Density    t={0:.3f}'.format(time))
#   ax.set_xlabel('X')
#   ax.set_ylabel('Y')
#   # ax.add_artist(circle1)
#   figName = 'dens_{0:03}.png'.format(nSnap)
#   if (particles): figName = 'dens_parts{0:03}.png'.format(nSnap)
#   fig.savefig( outDir + figName)
#
#   circle2 = plt.Circle((0.5, 0.5), r_interp, color='k', fill=False)
#
#   fig  = plt.figure(1)
#   fig.clf()
#   ax = plt.gca()
#   data = pot[cut,:,:]
#   img = ax.imshow( data, extent=[0,1,0,1] )
#   fig.colorbar( img )
#   ax.set_title('Potential    t={0:.3f}'.format(time))
#   ax.set_xlabel('X')
#   ax.set_ylabel('Y')
#   # ax.add_artist(circle2)
#   figName = 'pot_{0:03}.png'.format(nSnap)
#   if (particles): figName = 'pot_parts{0:03}.png'.format(nSnap)
#   fig.savefig( outDir + figName)
