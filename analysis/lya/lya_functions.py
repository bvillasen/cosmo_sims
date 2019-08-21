import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
from scipy import interpolate

dev_dir = '/home/bruno/Desktop/Dropbox/Developer/'
cosmo_dir = dev_dir + 'cosmo_sims/'
toolsDirectory = cosmo_dir + "tools/"
analysisDirectory = cosmo_dir + "analysis/"
sys.path.extend([toolsDirectory, analysisDirectory ] )
from tools import create_directory
from load_data_cholla import load_snapshot_data
from load_data_enzo import load_snapshot_enzo
from internal_energy import  get_temp, get_mu
# from cosmo_constants import *


#Boltazman constant
K_b = 1.38064852e-16 # g (cm/s)^2 K-1
#Mass of proton
M_p = 1.6726219e-24 #g
#Speed of ligth 
c = 2.99792458e10 #  cm/s
#Electron charge
e_charge = 4.8032e-10 # cm^3/2 g^1/2 s^-1 
#electron mass
M_e = 9.10938356e-28 #g
#Solar Mass
Msun = Msun = 1.98847e33  #g
#Parsec
pc = pc = 3.0857e18  #cm
kpc = 1000 * pc
Mpc = 1000 * kpc

#Doppler Shift for the freccuency
def get_nu( nu_0, v, c ):
 return nu_0 * ( 1 - v/c  )

def get_Doppler_parameter( T ):
  b = np.sqrt( 2* K_b / M_p * T )
  return b
  
def get_Doppler_width( nu_0, T ):
  b = get_Doppler_parameter( T ) 
  delta_nu = b / c * nu_0
  return delta_nu
   


def get_LOS_profiles( indx_i, indx_j, data, H0, Omega_M, Omega_L, Lx ): 
  cosmo_h = H0 / 100
  current_z_ch = data['current_z']
  dens = data['gas']['density'][...]
  temp = data['gas']['temperature'][...]
  dens_HI = data['gas']['HI_density'][...]
  vx = data['gas']['momentum_x'][...] / dens

  current_z = current_z_ch
  current_a = 1. / ( current_z + 1 )
  a_dot = np.sqrt( Omega_M/current_a + Omega_L*current_a**2  ) * H0 
  H = a_dot / current_a
  dens_los = dens[:, indx_j, indx_i]  
  dens_HI_los = dens_HI[:, indx_j, indx_i] 
  temp_los = temp[:, indx_j, indx_i]
  vel_los = vx[:, indx_j, indx_i] 
  
  nx = len( dens_los)

  x_comov = np.linspace( 0, Lx, nx )
  r_proper = current_a * x_comov / cosmo_h
  vel_Hubble = H * r_proper   #km/sec
  dr_proper = r_proper / nx


  #Convert to CGS Units
  dens_los *=  Msun / kpc**3 * cosmo_h*2
  dens_HI_los *=  Msun / kpc**3 * cosmo_h*2
  n_HI_los = dens_HI_los / M_p
  vel_los *= 1e5
  vel_Hubble *= 1e5
  
  nx_ref = 256
  dx_ref = Lx / nx_ref
  x_ref = dx_ref * ( np.linspace(0, nx_ref-1, nx_ref ) + 0.5 )
  r_intrp = current_a * x_ref / cosmo_h
  dens_intrp = np.interp( x_ref, x_comov, dens_los )
  dens_HI_intrp = np.interp( x_ref, x_comov, dens_HI_los )
  n_HI_intrp = np.interp( x_ref, x_comov, n_HI_los ) 
  vel_intrp = np.interp( x_ref, x_comov, vel_los )
  vel_Hubble_intrp = np.interp( x_ref, x_comov, vel_Hubble )
  temp_intrp = np.interp( x_ref, x_comov, temp_los )
  # return x_comov, r_proper, vel_Hubble, dens_los, dens_HI_los, n_HI_los, vel_los, temp_los  
  return x_ref, r_intrp, vel_Hubble_intrp, dens_intrp, dens_HI_intrp, n_HI_intrp, vel_intrp, temp_intrp  
  
def get_tau( Lya_nu, Lya_sigma, x_comov, N_HI_los, vel_Hubble, temp_los  ):
  nu = get_nu( Lya_nu, vel_Hubble, c)
  n_points = len( x_comov)
  tau_all = np.zeros(n_points)
  for i in range(n_points):
    nu_0 = nu[i]
    N_HI_0 = N_HI_los[i]
    temp_0 = temp_los[i]*20
    b = get_Doppler_parameter( temp_0 ) 
    delta_nu = get_Doppler_width( nu_0, temp_0) 
    exponent = ( nu - nu_0 ) / (delta_nu)  
    phi = 1 / ( np.sqrt(np.pi) * delta_nu ) * np.exp( -1 * exponent**2 )
    tau = Lya_sigma * N_HI_0 * phi
    tau_all += tau
  return tau_all