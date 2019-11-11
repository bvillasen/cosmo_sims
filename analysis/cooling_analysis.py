import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import yt

# dataDir = '/raid/bruno/data/cosmo_sims/'
dataDir = '/home/bruno/Desktop/data/'
cosmo_dir = '/home/bruno/Desktop/Dropbox/Developer/cosmo_sims/'
outDir = cosmo_dir + 'figures/phase_diagram/'
toolsDirectory = cosmo_dir + "tools/"
sys.path.extend([toolsDirectory ] )
from load_data_cholla import load_snapshot_data
from internal_energy import get_internal_energy, get_temp, get_Temperaure_From_Flags_DE
from cosmo_constants import *
# from load_data_enzo import load_snapshot_enzo_yt

# dataDir = '/home/bruno/Desktop/hdd_extrn_1/data/'
dataDir = '/raid/bruno/data/'
chollaDir = dataDir + 'cosmo_sims/cholla_pm/256_cool_uv_50Mpc/data_SIMPLE_PPMP_eta0.035_beta0.00_grav4/'
nSnap = 33

gamma = 5./3

# temp_0 = 1
# eta = 0.001
# 
# dens = 1.
# HI_dens = 0.76
# HII_dens = 0
# HeI_dens = .24
# HeII_dens = 0
# HeIII_dens = 0
# 
# mu =  dens / ( HI_dens + 2*HII_dens + ( HeI_dens + 2*HeII_dens + 3*HeIII_dens) / 4 )
# 
# u_0 = temp_0 / (gamma - 1) * K_b / M_p / mu
# v= np.sqrt( 2*u_0*(1/eta - 1))
# v_0 = 4.503  #km/sec
# 
# p_avrg = np.sqrt(3) * 370
# d_avrg = 13.49
# v_avrg = p_avrg / d_avrg
# v_avrg = 47.50 #km/sec 
# 
# 
# 
data_cholla = load_snapshot_data( nSnap, chollaDir, cool=True)
dens = data_cholla['gas']['density'][...]
dens_mean = dens.mean()
px = data_cholla['gas']['momentum_x'][...]
py = data_cholla['gas']['momentum_y'][...]
pz = data_cholla['gas']['momentum_z'][...]
temp = data_cholla['gas']['temperature'][...]
HI_dens = data_cholla['gas']['HI_density'][...]
HII_dens = data_cholla['gas']['HII_density'][...]
HeI_dens = data_cholla['gas']['HeI_density'][...]
HeII_dens = data_cholla['gas']['HeII_density'][...]
HeIII_dens = data_cholla['gas']['HeIII_density'][...]
e_dens = data_cholla['gas']['e_density'][...]
metal_dens = data_cholla['gas']['metal_density'][...]
GasEnergy = data_cholla['gas']['GasEnergy'][...]
Energy = data_cholla['gas']['Energy'][...]

frac_HI = HI_dens / dens
frac_HII = HII_dens / dens
frac_HeI = HeI_dens / dens
frac_HeII = HeII_dens / dens
frac_HeIII = HeIII_dens / dens
frac_e = e_dens / dens
frac_metal = metal_dens / dens



# vx = px / dens
# vy = py / dens
# vz = pz / dens
# Ekin = 0.5 * dens * ( vx*vx + vy*vy + vz*vz )
# U = Energy - Ekin
# indxs_U = np.where(flags_DE==0)
# indxs_ge = np.where(flags_DE==1)
# U_pressure = np.zeros_like( GasEnergy)
# U_pressure[indxs_U] = U[indxs_U]
# U_pressure[indxs_ge] = GasEnergy[indxs_ge]
# mu =  dens / ( HI_dens + 2*HII_dens + ( HeI_dens + 2*HeII_dens + 3*HeIII_dens) / 4 )
# temp_1 = get_temp( U_pressure/dens*1e6, gamma, mu )
# temp_U = temp_1[indxs_U].flatten()
# temp_GE = temp_1[indxs_ge].flatten()
# dens_U = dens[indxs_U].flatten() / dens_mean
# dens_GE = dens[indxs_ge].flatten() / dens_mean
# HI_dens_U = HI_dens[indxs_U].flatten() / dens_mean
# HI_dens_GE = HI_dens[indxs_ge].flatten() / dens_mean

# data_cholla = load_snapshot_data( nSnap, chollaDir, cool=True)
# current_z_ch = data_cholla['current_z']
# current_a_ch = data_cholla['current_a']
# dens = data_cholla['gas']['density'][...]
# px = data_cholla['gas']['momentum_x'][...]
# py = data_cholla['gas']['momentum_y'][...]
# pz = data_cholla['gas']['momentum_z'][...]
# temp = data_cholla['gas']['temperature'][...]
# flags_DE = data_cholla['gas']['flags_DE'][...]
# HI_dens = data_cholla['gas']['HI_density'][...]
# HII_dens = data_cholla['gas']['HII_density'][...]
# HeI_dens = data_cholla['gas']['HeI_density'][...]
# HeII_dens = data_cholla['gas']['HeII_density'][...]
# HeIII_dens = data_cholla['gas']['HeIII_density'][...]
# e_dens = data_cholla['gas']['e_density'][...]
# metal_dens = data_cholla['gas']['metal_density'][...]
# GasEnergy = data_cholla['gas']['GasEnergy'][...]
# Energy = data_cholla['gas']['Energy'][...]
# vx = px / dens
# vy = py / dens
# vz = pz / dens
# Ekin = 0.5 * dens * ( vx*vx + vy*vy + vz*vz )
# U = Energy - Ekin
# indxs_U = np.where(flags_DE==0)
# indxs_ge = np.where(flags_DE==1)
# U_pressure = np.zeros_like( GasEnergy)
# U_pressure[indxs_U] = U[indxs_U]
# U_pressure[indxs_ge] = GasEnergy[indxs_ge]
# 
# mu =  dens / ( HI_dens + 2*HII_dens + ( HeI_dens + 2*HeII_dens + 3*HeIII_dens) / 4 )
# temp_1 = get_temp( U_pressure/dens*1e6, gamma, mu )
# temp_2 = get_temp( GasEnergy/dens*1e6, gamma, mu )
# 



















# 
# snapKey = '{0:03}'.format(nSnap)
# inFileName = 'DD0{0}/data0{0}'.format( snapKey)
# 
# ds = yt.load( inDir + inFileName )
# data = ds.all_data()
# 
# h = ds.hubble_constant
# current_z = np.float(ds.current_redshift)
# current_a = 1./(current_z + 1)
# gamma = 5./3
# 
# data_grid = ds.covering_grid( level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions )
# gas_dens = data_grid[ ('gas', 'density')].in_units('msun/kpc**3').v*current_a**3/h**2
# # gas_vel_x = data_grid[('gas','velocity_x')].in_units('km/s').v
# # gas_vel_y = data_grid[('gas','velocity_y')].in_units('km/s').v
# # gas_vel_z = data_grid[('gas','velocity_z')].in_units('km/s').v
# gas_u = data_grid[('gas', 'thermal_energy' )].v * 1e-10  #km^2/s^2
# # gas_E = data_grid[('gas', 'total_energy' )].v * 1e-10   #km^2/s^2
# gas_temp = data_grid[ ('gas', 'temperature')].v
# 
# 
# HI_dens =  data_grid[ ('gas', 'H_p0_density')].in_units('msun/kpc**3').v*current_a**3/h**2
# HII_dens =  data_grid[ ('gas', 'H_p1_density')].in_units('msun/kpc**3').v*current_a**3/h**2
# HeI_dens =  data_grid[ ('gas', 'He_p0_density')].in_units('msun/kpc**3').v*current_a**3/h**2
# HeII_dens =  data_grid[ ('gas', 'He_p1_density')].in_units('msun/kpc**3').v*current_a**3/h**2
# HeIII_dens =  data_grid[ ('gas', 'He_p2_density')].in_units('msun/kpc**3').v*current_a**3/h**2
# electron_dens =  data_grid[ ('gas', 'El_density')].in_units('msun/kpc**3').v*current_a**3/h**2
# 
# mu_0 = data_grid[ ('gas', 'mean_molecular_weight')].v
# mu =  gas_dens / ( HI_dens + 2*HII_dens + ( HeI_dens + 2*HeII_dens + 3*HeIII_dens) / 4 )
# 
# diff = ( mu - mu_0 ) / mu_0
# 
# 




# metal_dens = data_grid[ ('gas', 'metal_density')].in_units('msun/kpc**3')*current_a**3/h**2
