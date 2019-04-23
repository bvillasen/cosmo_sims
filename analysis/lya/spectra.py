import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5

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


fileName = 'spectra_0.png'


f_12 = 1
lyman_constant = np.pi * e_charge**2 / M_e / c * f_12


nSnap = 13

# data_cholla = load_snapshot_data( str(nSnap), chollaDir, cool=True)
# current_z_ch = data_cholla['current_z']
# dens_ch = data_cholla['gas']['density'][...]
# temp_ch = data_cholla['gas']['temperature'][...]
# dens_H_ch = data_cholla['gas']['HI_density'][...]


data_enzo = load_snapshot_enzo( nSnap, enzoDir_uv, dm=False, cool=True)
current_z_en = data_enzo['current_z']
dens_en = data_enzo['gas']['density'][...]
temp_en =  data_enzo['gas']['temperature'][...]
dens_H_en = data_enzo['gas']['HI_density'][...]
vz_en = data_enzo['gas']['momentum_z'][...] / dens_en

# H0 = 67.8
# Omega_M = 0.308
# Omega_L = 1 - Omega_M
# cosmo_h = H0/100

H0 = 67.74                #[km/s / Mpc]
cosmo_h = H0/100
# H0 /= 1000                #[km/s / kpc]
Omega_M = 0.3089
Omega_L = 0.6911



lambda_0 = 1215.67e-10 #m     Lab wave length of the Lyman Alpha Transition

nu_0 = c / lambda_0

def get_nu( nu_0, v ):
 return nu_0 * ( 1 + v/c  )
 
 
 

 
 
 

current_z = current_z_en 
current_a = 1. / ( current_z + 1 )
a_dot = np.sqrt( Omega_M/current_a + Omega_L*current_a**2  ) * H0 
H = a_dot / current_a

L = 115. #Mpc
n_points = 256
dz = L / n_points
R = current_a * L / cosmo_h
dr = R / n_points

indx_j, indx_i = 128, 128

dens = dens_en[:, indx_j, indx_i]
dens_H = dens_H_en[:, indx_j, indx_i]
temp = temp_en[:, indx_j, indx_i]
vz = vz_en[:, indx_j, indx_i] * 1e3

x_comov = np.linspace( 0, L, n_points )
r_proper = current_a * x_comov / cosmo_h
vel = H * r_proper * 1e3  #m/sec




def get_integral( nu_center ):
  integral = 0
  print " "
  for i in range(n_points): 
    nH_cell = dens_H[i] / M_p * 1000
    v_cell = vel[i]
    nu_cell = get_nu( nu_0, v_cell )
    temp_cell = temp[i]
    b = np.sqrt( 2 * K_b * temp_cell / M_p )
    doppler_width = b / c * nu_cell
    integral_term = nH_cell / doppler_width / np.sqrt(np.pi) * np.exp( -( ( nu_cell - nu_center ) /doppler_width )**2  ) * dr
    print i, integral_term
    integral += integral_term  
  return integral

optical_depth = []

for n_cell in range(n_points):
  v_center = vel[n_cell]
  nu_center = get_nu( nu_0, v_center )
  integral = get_integral( nu_center )
  tau = lyman_constant * integral
  optical_depth.append(tau)
optical_depth = np.array( optical_depth )


ncols = 1
nrows = 4
fig, ax_l = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20*ncols,5*nrows))

plt.subplot(nrows, ncols, 1)
ax = plt.gca()
ax.clear()
ax.plot( x_comov, dens / dens.mean() , label='Gas' )
ax.plot( x_comov, dens_H / dens_H.mean()   , label='HI' )
ax.set_yscale('log')
ax.set_xlim(0, 115 )
ax.set_ylabel( r"$\rho / \bar{\rho} $", fontsize=16)
ax.legend( fontsize=16)

plt.subplot(nrows, ncols, 2)
ax = plt.gca()
ax.clear()
ax.plot( x_comov, temp  )
ax.set_yscale('log')
ax.set_xlim(0, 115 )
ax.set_ylim(3e3, 1e5 )
ax.set_ylabel( "Temperature [K]", fontsize=16)
ax.legend()

plt.subplot(nrows, ncols, 3)
ax = plt.gca()
ax.clear()
ax.plot( x_comov, vz  )
# ax.set_yscale('log')
ax.set_xlim(0, 115 )
ax.set_ylabel( r"LOS Velocity  [$km/s$] ", fontsize=16)
ax.legend()


plt.subplot(nrows, ncols, 4)
ax = plt.gca()
ax.clear()
ax.plot( x_comov, optical_depth )
ax.set_yscale('log')
ax.set_xlim(0, 115 )
ax.set_ylabel( r"$\tau$", fontsize=16)
ax.legend()



fig.tight_layout()
fig.savefig( outDir + fileName )
print 'Saved image: ', fileName
print ''

