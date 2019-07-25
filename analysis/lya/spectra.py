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

dataDir = '/raid/bruno/data/'
outDir = dev_dir + 'figures/spectra/'
create_directory( outDir )


eta_2 = 0.030

chollaDir = dataDir + 'cosmo_sims/cholla_pm/128_cool/data_PPMC_HLLC_SIMPLE_eta0.001_{0:.3f}_pressure/'.format(eta_2)
enzoDir_uv = dataDir + 'cosmo_sims/enzo/128_cool_uv/h5_files/'


# 
# #Boltazman constant
# K_b = 1.38064852e-23 #m2 kg s-2 K-1
# #Mass of proton
# M_p = 1.6726219e-27 #kg
# #Speed of ligth 
# c = 299792000.458 #  m/sec
# #Electron charge
# e_charge = 1.60217662e-19 # Coulombs 
# #electron mass
# M_e = 9.10938356e-31 #kg



#Boltazman constant
K_b = 1.38064852e-16 # erg K-1
#Mass of proton
M_p = 1.6726219e-24 #g
#Speed of ligth 
c = 2.99792458e10 #  cm/s
#Electron charge
e_charge = 4.8032e-10 # cm^3/2 g^1/2 s^-1 
#electron mass
M_e = 9.10938356e-28 #g



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
   
# Lymann Alpha Parameters
Lya_lambda = 1215.67e-10 #m     Rest wave length of the Lyman Alpha Transition
Lya_nu = c / Lya_lambda
f_12 = 0.416 #Oscillator strength
Lya_sigma = np.pi * e_charge**2 / M_e / c * f_12


#Test: Uniform Absorber
N_HI = 1e15    #cm^-2   Column density
T = 1e4 #K Temperature
nPoints = 1024
v_max = 100 #m/s
vel = np.linspace(-v_max, v_max, nPoints)
nu = get_nu( Lya_nu, vel, c)
b = get_Doppler_parameter( T ) 
delta_nu = get_Doppler_width( Lya_nu, T) 

phi = 1 / ( np.sqrt(np.pi) * delta_nu ) * np.exp( -1 * (( nu - Lya_nu ) / delta_nu  )**2 )
tau = Lya_sigma * N_HI * phi
vel_0 = vel[nPoints/2]
nu_0 = nu[nPoints/2]
phi_0 = phi[nPoints/2]
tau_0 = tau[nPoints/2]

b /= 1e3
tau_1 = 0.758 * ( N_HI / 1e13 ) * ( 10/ b )





Lbox = 50.0 #Mpc/h
nPoints = 128

Lx, Ly, Lz = Lbox, Lbox, Lbox
nx, ny, nz = nPoints, nPoints, nPoints
dx, dy, dz = Lx/nx, Ly/ny, Lz/nz
x = dx * ( np.linspace(0, nx-1, nx ) + 0.5 )
y = dy * ( np.linspace(0, ny-1, ny ) + 0.5 )
z = dz * ( np.linspace(0, nz-1, nz ) + 0.5 )



nSnap = 13
for nSnap in range(7, 28):
  outputFileName = 'spectra_{0}.png'.format(nSnap)

  data_cholla = load_snapshot_data( str(nSnap), chollaDir, cool=True)
  H0 = data_cholla['H0'] * 1e3 #[km/s / Mpc]
  cosmo_h = H0 / 100
  Omega_M = data_cholla['Omega_M']
  Omega_L = data_cholla['Omega_L']

  current_z_ch = data_cholla['current_z']
  dens_dm_ch = data_cholla['dm']['density'][...]
  dens_ch = data_cholla['gas']['density'][...]
  temp_ch = data_cholla['gas']['temperature'][...]
  dens_H_ch = data_cholla['gas']['HI_density'][...]
  vz_ch = data_cholla['gas']['momentum_z'][...] / dens_ch
  # dens_ch /= dens_ch.mean()
  # dens_H_ch /= dens_H_ch.mean()


  data_ch = [ dens_dm_ch, dens_ch, dens_H_ch, temp_ch, vz_ch ]


  data_enzo = load_snapshot_enzo( nSnap, enzoDir_uv, dm=True, cool=True)
  current_z_en = data_enzo['current_z']
  dens_dm_en = data_enzo['dm']['density'][...]
  dens_en = data_enzo['gas']['density'][...]
  temp_en =  data_enzo['gas']['temperature'][...]
  dens_H_en = data_enzo['gas']['HI_density'][...]
  vz_en = data_enzo['gas']['momentum_z'][...] / dens_en
  # dens_en /= dens_en.mean()
  # dens_H_en /= dens_H_en.mean()

  if current_z_ch > 15: temp_ch *= 0.97

  data_en = [ dens_dm_en, dens_en, dens_H_en, temp_en, vz_en ]

  data_all = [ data_en, data_ch ]

  ncols = 1
  nrows = 5
  fig, ax_l = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20*ncols,5*nrows))
  font_size = 25

  for n in range( 2 ):
    
    # if n == 0: continue
    
    
    data = data_all[n]
    dens_dm, dens, dens_H, temp, vz = data

    #Get Line of sight data
    indx_j, indx_i = 64, 64

    dens_dm = dens_dm[:, indx_j, indx_i]
    dens = dens[:, indx_j, indx_i]
    dens_H = dens_H[:, indx_j, indx_i]
    temp = temp[:, indx_j, indx_i]
    vz = vz[:, indx_j, indx_i] 


    #Interpolate data
    n_ref = 2
    nz_ref = nz * n_ref
    dz_ref = Lz / nz_ref
    z_ref = dz_ref * ( np.linspace(0, nz_ref-1, nz_ref ) + 0.5 )
    los_dens_dm = np.interp( z_ref, z, dens_dm ) 
    los_dens = np.interp( z_ref, z, dens )
    los_dens_H = np.interp( z_ref, z, dens_H)
    los_temp = np.interp( z_ref, z, temp )
    los_vel = np.interp( z_ref, z, vz )



    # Global Cosmological Parameters 
    current_z = current_z_ch 
    current_a = 1. / ( current_z + 1 )
    a_dot = np.sqrt( Omega_M/current_a + Omega_L*current_a**2  ) * H0 
    H = a_dot / current_a 

    # Get Proper coordinates
    Lz_proper = current_a * Lz / cosmo_h
    dz_proper = Lz / nz
    z_proper = current_a * z_ref / cosmo_h 


    if n == 0:
      # color = 'C0'
      ax_l[0].plot( z_ref, los_dens_dm  ,   linewidth=4, label='Enzo' )
      ax_l[1].plot( z_ref, los_dens  ,   linewidth=4 )
      ax_l[2].plot( z_ref, los_dens_H  ,   linewidth=4 )
      ax_l[3].plot( z_ref, los_temp, linewidth=4  )
      ax_l[4].plot( z_ref, los_vel, linewidth=4 )
      
      
      
    else:
      
      ax_l[0].plot( z_ref, los_dens_dm  , '--',  linewidth=4,  label='Cholla' )
      ax_l[1].plot( z_ref, los_dens  , '--',  linewidth=4 )
      ax_l[2].plot( z_ref, los_dens_H , '--',  linewidth=4 )
      ax_l[3].plot( z_ref, los_temp, '--',  linewidth=4)
      ax_l[4].plot( z_ref, los_vel, '--', linewidth=4 )
      





  ax = ax_l[0]
  ax.set_title(r'z={0:.2f}   $\eta_2={1:0.3f}$   LOS Comparison '.format( current_z_ch, eta_2,  ), fontsize=25, )
  ax.set_yscale('log')
  ax.set_xlim(0, Lz )
  ax.set_ylabel( r"$\rho_{DM}$", fontsize=font_size)
  ax.legend( fontsize = font_size)

  ax = ax_l[1]
  ax.set_yscale('log')
  ax.set_xlim(0, Lz )
  ax.set_ylabel( r"$\rho_{GAS}$", fontsize=font_size)


  ax = ax_l[2]
  ax.set_yscale('log')
  ax.set_xlim(0, Lz )
  ax.set_ylabel( r"$\rho_{HI}$", fontsize=font_size)



  ax = ax_l[3]
  ax.set_yscale('log')
  ax.set_xlim(0, Lz )
  ax.set_ylabel( "Temperature [K]", fontsize=font_size)
  if current_z_ch > 15:
    ax.set_ylim(1, 400 )
  else:
    ax.set_ylim(600, 40000 )
  

  ax = ax_l[4]
  # ax.set_yscale('log')
  ax.set_xlim(0, Lz )
  ax.set_ylabel( r"LOS Velocity  [$km/s$] ", fontsize=font_size)
  # ax.legend()
  # 
  # ax = ax_l[3]
  # # ax.plot( x_comov, optical_depth )
  # ax.set_yscale('log')
  # ax.set_xlim(0, Lz )
  # ax.set_ylabel( r"$\tau$", fontsize=font_size)
  # # ax.legend()

  ax.set_xlabel(" X [ Mpc/h]", fontsize=font_size)


  fig.tight_layout()
  fig.savefig( outDir + outputFileName )
  print 'Saved image: ', outputFileName
  print ''
