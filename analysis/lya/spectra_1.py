import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import matplotlib as mpl

mpl.rcParams['axes.linewidth'] = 3 #set the value globally
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['xtick.major.size'] = 10
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['ytick.major.size'] = 10
cosmo_dir = '/home/bruno/Desktop/Dropbox/Developer/cosmo_sims/'
toolsDirectory = cosmo_dir + "tools/"
analysisDirectory = cosmo_dir + "analysis/"
sys.path.extend([toolsDirectory, analysisDirectory ] )
from load_data_cholla import load_snapshot_data
from load_data_enzo import load_snapshot_enzo
from internal_energy import  get_temp, get_mu
from cosmo_constants import *

dataDir = '/raid/bruno/data/'
outDir = cosmo_dir + 'figures/spectra/'

chollaDir = dataDir + 'cosmo_sims/cholla_pm/256_cool/data_de02_limitEkin/'
enzoDir_uv = dataDir + 'cosmo_sims/enzo/256_cool_uv/h5_files/'




# f_12 = 1
# lyman_constant = np.pi * e_charge**2 / M_e / c * f_12

sigma_0 = 4.5e-22 #m^2
lyman_constant = sigma_0 * c / np.sqrt(np.pi)

# data = 'cholla'
# data = 'enzo'

nSnap = 13


L = 115. #Mpc
n_points = 256
dz = L / n_points


H0 = 67.74                #[km/s / Mpc]
cosmo_h = H0/100
# H0 /= 1000                #[km/s / kpc]
Omega_M = 0.3089
Omega_L = 0.6911

lambda_0 = 1215.67e-10 #m     Lab wave length of the Lyman Alpha Transition

nu_0 = c / lambda_0

def get_nu( nu_0, v ):
 return nu_0 * ( 1 + v/c  )
 
fileName = 'spectra_2.png'
 

def get_integral( v_center, dens_H, vel, temp ):
  integral = 0
  # print " "
  for i in range(n_points): 
    nH_cell = dens_H[i] / M_p 
    v_cell = vel[i]
    temp_cell = temp[i]
    b = np.sqrt( 2 * K_b * temp_cell / M_p )*5
    integral_term = nH_cell / b  * np.exp( -( ( v_cell - v_center ) /b )**2  ) * dr * Mpc
    # print i, integral_term
    integral += integral_term  
  return integral

def get_optical_depth( dens, dens_H, temp, vel, vz  ):
  optical_depth = []

  for n_cell in range(n_points):
    v_center = vel[n_cell]
    integral = get_integral( v_center, dens_H, vel, temp )
    tau = lyman_constant * integral
    optical_depth.append(tau)
    
  optical_depth = np.array( optical_depth ) * 20
  F = np.exp(-optical_depth)
  return optical_depth, F


data_cholla = load_snapshot_data( str(nSnap), chollaDir, cool=True)
current_z = data_cholla['current_z']
dens = data_cholla['gas']['density'][...]
temp = data_cholla['gas']['temperature'][...]
dens_H = data_cholla['gas']['HI_density'][...]
vz = data_cholla['gas']['momentum_z'][...] / dens
indx_j, indx_i = 128, 128

current_z = current_z
current_a = 1. / ( current_z + 1 )
a_dot = np.sqrt( Omega_M/current_a + Omega_L*current_a**2  ) * H0 
H = a_dot / current_a
R = current_a * L / cosmo_h
dr = R / n_points
dens_ch = dens[:, indx_j, indx_i]  * Msun / kpc**3 * cosmo_h*2
dens_H_ch = dens_H[:, indx_j, indx_i] * Msun / kpc**3 * cosmo_h*2
temp_ch = temp[:, indx_j, indx_i]
vz_ch = vz[:, indx_j, indx_i] * 1e3

x_comov = np.linspace( 0, L, n_points )
r_proper = current_a * x_comov / cosmo_h
vel = H * r_proper * 1e3  #m/sec

optical_depth_ch, F_ch = get_optical_depth( dens_ch, dens_H_ch, temp_ch, vel, vz_ch  )


# if data == 'enzo':
data_enzo = load_snapshot_enzo( nSnap, enzoDir_uv, dm=False, cool=True)
current_z = data_enzo['current_z']
dens = data_enzo['gas']['density'][...]
temp =  data_enzo['gas']['temperature'][...]
dens_H = data_enzo['gas']['HI_density'][...]
vz = data_enzo['gas']['momentum_z'][...] / dens
indx_j, indx_i = 128, 128

current_z = current_z
current_a = 1. / ( current_z + 1 )
a_dot = np.sqrt( Omega_M/current_a + Omega_L*current_a**2  ) * H0 
H = a_dot / current_a
R = current_a * L / cosmo_h
dr = R / n_points
dens_en = dens[:, indx_j, indx_i]  * Msun / kpc**3 * cosmo_h*2
dens_H_en = dens_H[:, indx_j, indx_i] * Msun / kpc**3 * cosmo_h*2
temp_en = temp[:, indx_j, indx_i]
vz_en = vz[:, indx_j, indx_i] * 1e3

x_comov = np.linspace( 0, L, n_points )
r_proper = current_a * x_comov / cosmo_h
vel = H * r_proper * 1e3  #m/sec

optical_depth_en, F_en = get_optical_depth( dens_en, dens_H_en, temp_en, vel, vz_en  )












ncols = 1
nrows = 5
fig, ax_l = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20*ncols,5*nrows))
fsize = 22

plt.subplot(nrows, ncols, 1)
ax = plt.gca()
ax.clear()
# ax.plot( x_comov, dens_en / dens_en.mean() , label='Gas', linewidth=4 )
ax.plot( x_comov, dens_H_en  , label='HI_enzo', linewidth=4 )
ax.plot( x_comov, dens_H_ch  , label='HI_cholla', linewidth=4 )
ax.set_yscale('log')
ax.set_xlim(0, 115 )
# ax.set_ylim(0.01, 6e2 )
ax.set_ylabel( r"$\rho$  [$h^2\mathrm{M}_{\odot} / \mathrm{kpc}^3$] ", fontsize=fsize)
ax.legend( fontsize=fsize)
ax.xaxis.tick_top()
ax.set_title(r'Comoving Distance [$Mpc/h$]', fontsize=fsize, pad = 50)
ax.xaxis.label_position ='top'

# plt.subplot(nrows, ncols, 2)
# ax = plt.gca()
# ax.clear()
# ax.plot( x_comov, dens_ch / dens_ch.mean() , label='Gas', linewidth=4 )
# ax.plot( x_comov, dens_H_ch / dens_H_ch.mean()   , label='HI', linewidth=4 )
# ax.set_yscale('log')
# ax.set_xlim(0, 115 )
# ax.set_ylim(0.01, 6e2 )
# # ax.set_ylabel( r"$\rho / \bar{\rho} $", fontsize=fsize)
# ax.legend( fontsize=fsize)
# ax.xaxis.tick_top()
# ax.set_title(r'Comoving Distance [$Mpc/h$]', fontsize=fsize, pad = 50)
# ax.xaxis.label_position ='top'


plt.subplot(nrows, ncols, 2)
ax = plt.gca()
ax.clear()
ax.plot( x_comov, temp_en, label='Enzo', linewidth=4   )
ax.plot( x_comov, temp_ch, label='Cholla', linewidth=4   )
ax.set_yscale('log')
ax.set_xlim(0, 115 )
ax.set_ylim(3e3, 2e5 )
ax.set_ylabel( "Temperature [K]", fontsize=fsize)
ax.legend( fontsize=fsize)

# plt.subplot(nrows, ncols, 4)
# ax = plt.gca()
# ax.clear()
# ax.plot( x_comov, temp_ch, linewidth=4   )
# ax.set_yscale('log')
# ax.set_xlim(0, 115 )
# ax.set_ylim(3e3, 2e5 )
# # ax.set_ylabel( "Temperature [K]", fontsize=fsize)
# ax.legend()

plt.subplot(nrows, ncols, 3)
ax = plt.gca()
ax.clear()
ax.plot( x_comov, vz_en * 1e-3, linewidth=4, label='Enzo'   )
ax.plot( x_comov, vz_ch * 1e-3, linewidth=4, label='Cholla'   )
# ax.set_yscale('log')
ax.set_xlim(0, 115 )
ax.set_ylim(-2e2, 5e2 )
ax.set_ylabel( r"LOS Velocity  [$km/s$] ", fontsize=fsize)
ax.legend( fontsize=fsize)

# plt.subplot(nrows, ncols, 6)
# ax = plt.gca()
# ax.clear()
# ax.plot( x_comov, vz_ch * 1e-3, linewidth=4   )
# # ax.set_yscale('log')
# ax.set_xlim(0, 115 )
# ax.set_ylim(-2e2, 5e2 )
# # ax.set_ylabel( r"LOS Velocity  [$km/s$] ", fontsize=fsize)
# ax.legend()


plt.subplot(nrows, ncols, 4)
ax = plt.gca()
ax.clear()
ax.plot( x_comov, optical_depth_en, linewidth=4, label="Enzo"   )
ax.plot( x_comov, optical_depth_ch, linewidth=4, label="Cholla"  )

ax.set_yscale('log')
ax.set_xlim(0, 115 )
ax.set_ylim(1e-2, 1e2 )
ax.set_ylabel( r"$\tau$", fontsize=fsize)
ax.legend( fontsize=fsize)

# plt.subplot(nrows, ncols, 8)
# ax = plt.gca()
# ax.clear()
# ax.plot( x_comov, optical_depth_ch, linewidth=4  )
# ax.set_yscale('log')
# ax.set_xlim(0, 115 )
# ax.set_ylim(1e-2, 1e2 )
# ax.set_ylabel( r"$\tau$", fontsize=fsize)
# ax.legend()

plt.subplot(nrows, ncols, 5)
ax = plt.gca()
ax.clear()
ax.plot( vel *1e-3, F_en, linewidth=4, label="Enzo"  )
ax.plot( vel *1e-3, F_ch, linewidth=4, label="Cholla"  )
ax.set_xlim(0, (vel *1e-3).max() )
ax.set_ylim(0, 1 )
ax.set_ylabel( r"$F$", fontsize=fsize)
ax.set_xlabel( r"Velocity [$km/s$]", fontsize=fsize)
ax.legend( fontsize=fsize)

# plt.subplot(nrows, ncols, 10)
# ax = plt.gca()
# ax.clear()
# ax.plot( vel *1e-3, F_ch, linewidth=4  )
# ax.set_xlim(0, (vel *1e-3).max() )
# ax.set_ylim(0, 1 )
# ax.set_ylabel( r"$F$", fontsize=fsize)
# ax.set_xlabel( r"Velocity [$km/s$]", fontsize=fsize)
# ax.legend()



fig.tight_layout()
fig.savefig( outDir + fileName )
print 'Saved image: ', fileName
print ''

# 