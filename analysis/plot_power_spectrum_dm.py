import sys
import numpy as np
import matplotlib.pyplot as plt
# from np.fft import fftn
# from colossus.cosmology import cosmology
from scipy.interpolate import RegularGridInterpolator
import h5py as h5
from power_spectrum import get_power_spectrum

dataDir = '/raid/bruno/data/'
# dataDir = '/home/bruno/Desktop/hard_drive_1/data/'
cosmo_dir = '/home/bruno/Desktop/Dropbox/Developer/cosmo_sims/'
toolsDirectory = cosmo_dir + "tools/"
sys.path.extend([toolsDirectory ] )
from load_data_cholla import load_snapshot_data_particles
from load_data_nyx import load_snapshot_nyx

nyxDir = dataDir + 'cosmo_sims/nyx/256_dm/'
chollaDir = dataDir + 'cosmo_sims/cholla_pm/256_dm/data/'
outDir = cosmo_dir + 'figures/power_dm/'

# snapshots_ch = [  2, 4, 6, 8, 10, 12, 16, 20, 24, 30 ]
snapshots = range(20)

fileName = 'power_dm_nyx_50Mpc.png'


# Lbox = 115.0   #Mpc/h
# h = 0.6774

Lbox = 50.
h = 0.6766

nPoints = 256
nz, ny, nx = nPoints, nPoints, nPoints
nCells  = nx*ny*nz
Lx = Lbox
Ly = Lbox
Lz = Lbox
dx, dy, dz = Lx/(nx), Ly/(ny), Lz/(nz )
n_kSamples = 12

fig = plt.figure(0)
fig.set_size_inches(10,12)
fig.clf()


colors = ['b', 'y', 'g', 'c', 'm', 'b', 'y', 'g', 'c', 'm','m', 'b', 'y', 'g', 'c', 'm','m', 'b', 'y', 'g', 'c', 'm' ]

gs = plt.GridSpec(5, 1)
gs.update(hspace=0.05)
ax1 = plt.subplot(gs[0:4, 0])
ax2 = plt.subplot(gs[4:5, 0])

for i,nSnap in enumerate(snapshots):
  print 'Snapshot: ', nSnap
  # if i not in [1,]: continue
  snapKey = str(nSnap)
  c = colors[i]
  # Load cholla densities
  data_cholla = load_snapshot_data_particles( nSnap, chollaDir )
  current_z_ch_1 = data_cholla['current_z']
  dens_dm_cholla_1 = data_cholla['density'][...]
  print " Cholla: ", current_z_ch_1, dens_dm_cholla_1.mean()

  ps_dm_cholla, k_vals, error_cholla = get_power_spectrum( dens_dm_cholla_1, Lbox, nx, ny, nz, dx, dy, dz,  n_kSamples=n_kSamples)

  #Load Nyx data
  print ' Nyx: ', nSnap
  data_nyx = load_snapshot_nyx( nSnap, nyxDir, hydro=False )
  current_a_nyx = data_nyx['current_a']
  current_z_nyx = data_nyx['current_z']
  print current_z_nyx
  dens_dm_nyx = data_nyx['dm']['density'][...]

  ps_dm_nyx, k_vals, count_dm_nyx = get_power_spectrum( dens_dm_nyx, Lbox, nx, ny, nz, dx, dy, dz,  n_kSamples=n_kSamples)

  error = ( ps_dm_cholla - ps_dm_nyx ) /ps_dm_cholla

  print "Error: {0}\n".format(error.max())


  print "Plotting..."
  label = 'z = {0:.1f}'.format(current_z_ch_1)
  ax1.plot( k_vals, ps_dm_cholla, label=label, c=c, linewidth=2 )
  # plt.errorbar( k_vals, ps_dm_cholla, yerr=error_cholla, color='g')
  if i==len(snapshots)-1: ax1.plot( k_vals, ps_dm_nyx, '--', c='k', label='Nyx' )
  else: ax1.plot( k_vals, ps_dm_nyx, '--', c='k' )
  # plt.errorbar( k_vals, ps_dm_enzo, yerr=error_enzo, color='r')
  # plt.plot( k_vals, ps_dm_gad, '--', c='r', alpha=0.7 )
  # plt.plot( k_vals, power_spectrum * growthFactor**2, c='k' )

  ax2.plot( k_vals, error, c='r', alpha=0.7 )


#
ax2.axhline( y=0., color='r', linestyle='--',  )
ax2.set_ylim( -0.02, 0.02 )

ax1.set_ylabel( r'$P(k) $', fontsize=17)
ax2.set_ylabel( 'Difference', fontsize=15)
ax2.set_xlabel( r'$k \, \, [h Mpc^-1]$', fontsize=17)
ax1.legend( loc=3)
ax1.set_xscale('log')
ax1.set_yscale('log')

ax2.set_xscale('log')
# plt.xlim()
ax1.set_title( 'DM Power Spectrum   {0}', fontsize=20)
print "Saved File: ", fileName
fig.savefig( outDir + fileName,  pad_inches=0.1,  bbox_inches='tight', dpi=300)
# # plt.show()
#
#
# inFile.close()




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
