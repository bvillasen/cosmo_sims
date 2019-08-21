import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
from scipy import interpolate
import matplotlib.gridspec as gridspec
import matplotlib as mpl


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
from lya_functions import *

dataDir = '/raid/bruno/data/'
outDir = dev_dir + 'figures/spectra/'
create_directory( outDir )


eta_2 = 0.050

chollaDir = dataDir + 'cosmo_sims/cholla_pm/128_cool/data_PPMC_HLLC_SIMPLE_eta0.001_{0:.4f}/'.format(eta_2)
enzoDir_uv = dataDir + 'cosmo_sims/enzo/128_cool_uv/h5_files/'

# chollaDir = dataDir + 'cosmo_sims/cholla_pm/256_cool_uv_100Mpc/data_PPMC_HLLC_SIMPLE_eta0.001_0.034_reconstDE/'
# enzoDir_uv = dataDir + 'cosmo_sims/enzo/256_cool_uv_100Mpc/h5_files/'




# Lymann Alpha Parameters
Lya_lambda = 1215.67e-10 #m     Rest wave length of the Lyman Alpha Transition
Lya_lambda = 1215.67e-8 #cm
Lya_nu = c / Lya_lambda
f_12 = 0.416 #Oscillator strength
Lya_sigma = np.pi * e_charge**2 / M_e / c * f_12





Lbox = 50.0 #Mpc/h
nPoints = 128

Lx, Ly, Lz = Lbox, Lbox, Lbox
nx, ny, nz = nPoints, nPoints, nPoints
dx, dy, dz = Lx/nx, Ly/ny, Lz/nz
x = dx * ( np.linspace(0, nx-1, nx ) + 0.5 )
y = dy * ( np.linspace(0, ny-1, ny ) + 0.5 )
z = dz * ( np.linspace(0, nz-1, nz ) + 0.5 )


indx_j, indx_i = 64, 64
# 
nSnap = 12
# for nSnap in range(7, 28):
# outputFileName = 'spectra_{0}_{1:.4f}.png'.format(nSnap, eta_2)
outputFileName = 'spectra_{0}.png'.format(nSnap, eta_2)

data_cholla = load_snapshot_data( str(nSnap), chollaDir, cool=True)
H0 = data_cholla['H0'] * 1e3 #[km/s / Mpc]
cosmo_h = H0 / 100
Omega_M = data_cholla['Omega_M']
Omega_L = data_cholla['Omega_L']
current_a = data_cholla['current_a']
current_z = data_cholla['current_z']
R = current_a * Lx / cosmo_h
dr = R / nx
dr_cm = dr * Mpc

data_enzo = load_snapshot_enzo( nSnap, enzoDir_uv, dm=False, cool=True)




x_comov, r_proper, vel_Hubble, dens_los, dens_HI_los, n_HI_los, vel_los, temp_los = get_LOS_profiles( indx_i, indx_j, data_cholla, H0, Omega_M, Omega_L, Lx )
N_HI_los = n_HI_los * dr_cm 
tau_all = get_tau( Lya_nu, Lya_sigma, x_comov, N_HI_los, vel_Hubble, temp_los,  )
tau_all *= 5
F = np.exp(-tau_all)
data_ch = [ N_HI_los,  temp_los, vel_los, tau_all, F ]

x_comov, r_proper, vel_Hubble, dens_los, dens_HI_los, n_HI_los, vel_los, temp_los = get_LOS_profiles( indx_i, indx_j, data_enzo, H0, Omega_M, Omega_L, Lx )
N_HI_los = n_HI_los * dr_cm 
tau_all = get_tau( Lya_nu, Lya_sigma, x_comov, N_HI_los, vel_Hubble, temp_los,  )
tau_all *= 5
F = np.exp(-tau_all)
data_en = [ N_HI_los, temp_los, vel_los, tau_all, F ]

vel_Hubble *= 1e-5


mpl.rcParams['axes.linewidth'] = 4 #set the value globally


tick_size_0 = 20
tick_size_1 = 20
c_en = 'C0'
c_ch = 'C1'
font_size = 35
line_width_1 = 6
line_width = 5
ncols = 1
nrows = 5
# fig, ax_l = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20*ncols,5*nrows))\
fig = plt.figure( figsize=(20*ncols,5*nrows) )
gs1 = gridspec.GridSpec(nrows,1)
gs1.update(wspace=0, hspace=0) # set the spacing between axes. 


# ax = ax_l[0]
ax = plt.subplot(gs1[0])
ax.set_title( r"Simulated Ly-$\alpha$ Forest Spectra    z={0:.2f}".format(current_z), fontsize=font_size)
ax.plot( x_comov, data_en[0] , linewidth=line_width_1, c=c_en, label="ENZO")
ax.plot( x_comov, data_ch[0] , '--',linewidth=line_width, c=c_ch, label="CHOLLA")
ax.legend(fontsize=30)
ax.set_yscale('log')
ax.set_xlim( x_comov.min(), x_comov.max())
ax.set_ylabel( r'$N_{HI}  \,\,\, [cm^{-2}]$ ', fontsize=font_size)
ax.tick_params(axis='both', which='major', labelsize=tick_size_0)
ax.tick_params(axis='both', which='minor', labelsize=tick_size_1)

# ax = ax_l[1]
ax = plt.subplot(gs1[1])
ax.plot( vel_Hubble, data_en[1], linewidth=line_width_1, c=c_en)
ax.plot( vel_Hubble, data_ch[1],'--', linewidth=line_width, c=c_ch)
ax.set_yscale('log')
ax.set_xlim( vel_Hubble.min(), vel_Hubble.max())
ax.set_ylabel( r'$T \,\,\,[K]$ ', fontsize=font_size)
ax.tick_params(axis='both', which='major', labelsize=tick_size_0)
ax.tick_params(axis='both', which='minor', labelsize=tick_size_1)

# ax = ax_l[2]
ax = plt.subplot(gs1[2])
ax.plot( vel_Hubble, data_en[2]*1e-5, linewidth=line_width_1, c=c_en)
ax.plot( vel_Hubble, data_ch[2]*1e-5, '--',linewidth=line_width, c=c_ch)
# ax.set_yscale('log')
ax.set_xlim( vel_Hubble.min(), vel_Hubble.max())
ax.set_ylabel( r'$v_{los} \,\,\, [km/s]$ ', fontsize=font_size)
ax.tick_params(axis='both', which='major', labelsize=tick_size_0)
ax.tick_params(axis='both', which='minor', labelsize=tick_size_1)

# ax = ax_l[3]
ax = plt.subplot(gs1[3])
ax.plot( vel_Hubble, data_en[3], linewidth=line_width_1, c=c_en)
ax.plot( vel_Hubble, data_ch[3], '--', linewidth=line_width, c=c_ch)
ax.set_yscale('log')
ax.set_xlim( vel_Hubble.min(), vel_Hubble.max())
ax.set_ylabel( r'$\tau$ ', fontsize=font_size)
ax.tick_params(axis='both', which='major', labelsize=tick_size_0)
ax.tick_params(axis='both', which='minor', labelsize=tick_size_1)

# ax = ax_l[4]
ax = plt.subplot(gs1[4])
ax.plot( vel_Hubble, data_en[4], linewidth=line_width_1, c=c_en)
ax.plot( vel_Hubble, data_ch[4], '--',linewidth=line_width, c=c_ch)
ax.set_xlim( vel_Hubble.min(), vel_Hubble.max())
ax.set_ylim( 0, 1 )
ax.set_ylabel( r'$F$ ', fontsize=font_size)
ax.set_xlabel( r'$v \,\,\,  [km / s]$', fontsize=font_size)
ax.tick_params(axis='both', which='major', labelsize=tick_size_0)
ax.tick_params(axis='both', which='minor', labelsize=tick_size_1)

fig.subplots_adjust( wspace=0 )
fig.tight_layout()
fig.savefig( outDir + outputFileName )
print 'Saved image: ', outputFileName
print ''










