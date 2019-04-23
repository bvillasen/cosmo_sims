import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5

cosmo_dir = '/home/bruno/Desktop/Dropbox/Developer/cosmo_sims/'
outDir = cosmo_dir + 'figures/phase_diagram/'
toolsDirectory = cosmo_dir + "tools/"
sys.path.extend([toolsDirectory ] )
from load_data_cholla import load_snapshot_data
from load_data_enzo import load_snapshot_enzo
from phase_diagram import get_phase_diagram
from internal_energy import get_internal_energy, get_temp, get_Temperaure_From_Flags_DE, get_mu

dataDir = '/raid/bruno/data/'
outDir = cosmo_dir + 'figures/error_cells/'

chollaDir_0 = dataDir + 'cosmo_sims/cholla_pm/256_cool/data_de02_error_cells/'
# chollaDir_1 = dataDir + 'cosmo_sims/cholla_pm/256_cool/data_de02_limitChangeDE/'
# chollaDir_2 = dataDir + 'cosmo_sims/cholla_pm/256_cool/data_de02_grav2/'


enzoDir_uv = dataDir + 'cosmo_sims/enzo/256_cool_uv/h5_files/'


gamma = 5./3
nPoints = 256
nx = nPoints
ny = nPoints
nz = nPoints
ncells = nx * ny * nz

dv = (115000./nPoints)**3
nbins = 1000


err_indxs_tr = (np.array([ 67,  92, 160, 161, 171]),
 np.array([242,  28,  66,  10, 192]),
 np.array([190, 190, 100, 240, 176]))
 
n_err = len(err_indxs_tr[0])
err_indxs = []
for i in range(n_err):
  k = err_indxs_tr[0][i]
  j = err_indxs_tr[1][i]
  i = err_indxs_tr[2][i]
  err_indx = ( np.array([k]), np.array([j]), np.array([i]) )
  err_indxs.append( err_indx )


H0 = 67.74     #[km/s / Mpc]
cosmo_h = H0/100
H0 /= 1000      #[km/s / kpc]
Omega_M = 0.3089
Omega_L = 0.6911
Omega_K = 0.0
cosmo_G = 4.300927161e-06; # gravitational constant, kpc km^2 s^-2 Msun^-1

r_0_gas = 1.0
rho_0_gas = 3*H0*H0 / ( 8*np.pi*cosmo_G ) * Omega_M /cosmo_h/cosmo_h
t_0_gas = 1/H0*cosmo_h
v_0_gas = r_0_gas / t_0_gas
phi_0_gas = v_0_gas * v_0_gas





ncols = 1
nrows = 5
fig, ax_l = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10,8))

x_min = 10
x_max = 28
y_min = -1000
y_max = 3000

index = 1
for index in range( n_err):
  err_indx = err_indxs[index]


  ge_0_list = []
  ge_1_list = []
  adv_term_list = []
  p_term_list = []
  ge_next_list = []
  x_0 = []
  x_1 = []
  temp_list = []

  # 
  nSnap = 27
  # for nSnap in range(10, 27):

  k0, j0, i0 = err_indx 
  k0 = k0[0]
  j0 = j0[0]
  i0 = i0[0]
  k_list, j_list, i_list = [], [], []
  for k in range(-1,2,2):
    for j in range(-1,2,2):
      for i in range(-1,2,2):
        k_list.append( k0 + k )
        j_list.append( j0 + j )
        i_list.append( i0 + i )

  arr_neig = ( np.array(k_list), np.array(j_list), np.array(i_list),) 
  data_cholla = load_snapshot_data( str(nSnap), chollaDir_0, cool=True)
  current_z_ch = data_cholla['current_z']
  current_a_ch = data_cholla['current_a']
  current_a = current_a_ch - 0.001
  dt = data_cholla['dt'] * t_0_gas
  dens = data_cholla['gas']['density'][...]
  ge = data_cholla['gas']['GasEnergy'][...]
  flags = data_cholla['gas']['flags_DE'][...]
  # vx = data_cholla['gas']['momentum_x'][...] / dens
  # vy = data_cholla['gas']['momentum_y'][...] / dens
  # vz = data_cholla['gas']['momentum_z'][...] / dens
  # temp_GK, temp_DE, temp_U, temp_GE, dens_U, dens_GE, HI_dens_U, HI_dens_GE, temp_U_ALL, temp_GE_ALL =  get_Temperaure_From_Flags_DE( data_cholla, gamma=5./3, normalize_dens=True, )
  mu = get_mu(data_cholla)
  temps = get_temp( ge / dens * 1e6, mu=mu )

  # adv_terms = data_cholla['gas']['extra_scalar'][...] * rho_0_gas * phi_0_gas / current_a / current_a
  # p_terms = data_cholla['gas']['extra_scalar_1'][...] * rho_0_gas * phi_0_gas / current_a / current_a

  flags_0 = flags[arr_neig]
  print flags_0

  # d_0 = dens[err_indx][0]
  # vx_0 = vx[err_indx][0]
  # vy_0 = vy[err_indx][0]
  # vz_0 = vz[err_indx][0]
  ge_0 = ge[err_indx][0]
  temp_0 = temps[err_indx][0]
  # mu_0 = mu[err_indx][0]

  # 
  # ge_0_list.append(ge_0)
  # x_0.append(nSnap)
  # temp_list.append( max(temp_0, 1))
  # 
  # # if temp_0 <= 1: break
  # 
  # 


  # nSnap += 1
  # 
  # data_cholla = load_snapshot_data( str(nSnap), chollaDir_0, cool=True)
  # current_z_ch = data_cholla['current_z']
  # current_a_ch = data_cholla['current_a']
  # current_a = current_a_ch - 0.001
  # dt = data_cholla['dt'] * t_0_gas
  # 
  # # dens = data_cholla['gas']['density'][...]
  # ge = data_cholla['gas']['GasEnergy'][...]
  # # vx = data_cholla['gas']['momentum_x'][...] / dens
  # # vy = data_cholla['gas']['momentum_y'][...] / dens
  # # vz = data_cholla['gas']['momentum_z'][...] / dens
  # # temp_GK, temp_DE, temp_U, temp_GE, dens_U, dens_GE, HI_dens_U, HI_dens_GE, temp_U_ALL, temp_GE_ALL =  get_Temperaure_From_Flags_DE( data_cholla, gamma=5./3, normalize_dens=True, )
  # # mu = get_mu(data_cholla)
  # # temps = get_temp( ge / dens * 1e6, mu=mu )
  # 
  # # p_terms = data_cholla['gas']['extra_scalar'][...]
  # adv_terms = data_cholla['gas']['extra_scalar'][...] * rho_0_gas * phi_0_gas / current_a / current_a
  # p_terms = data_cholla['gas']['extra_scalar_1'][...] * rho_0_gas * phi_0_gas / current_a / current_a
  # adv_term_1 = adv_terms[err_indx][0]
  # p_term_1 = p_terms[err_indx][0]
  # 
  # 
  # ge_next = ge_0 + adv_term_1 + p_term_1
  # 
  # 
  # d_1 = dens[err_indx][0]
  # # vx_1 = vx[err_indx][0]
  # # vy_1 = vy[err_indx][0]
  # # vz_1 = vz[err_indx][0]
  # # temp_1 = temps[err_indx][0]
  # ge_1 = ge[err_indx][0]
  # 
  # 
  # ge_1_list.append(ge_1)
  # p_term_list.append( p_term_1)
  # adv_term_list.append( adv_term_1 )
  # ge_next_list.append( ge_next )
  # x_1.append(nSnap)

#   x_0 = np.array( x_0 )
#   x_1 = np.array( x_1 )
# 
#   y_max = max(ge_0_list)* 1.2
#   y_min = min( adv_term_list) * 1.5
# 
#   plt.subplot(nrows, ncols, index+1)
#   ax = plt.gca()
#   ax.plot(x_0, ge_0_list, 'o', label="U_advected" )
#   ax.plot(x_1-1, adv_term_list, 'o', label=r'$\Delta$ advc_term' )
#   ax.plot(x_1-1, p_term_list, 'o', label=r'$\Delta$ P Div(v)' )
#   # ax.plot(x_0+1, ge_1_list, 'o', markersize=4 )
#   ax.plot(x_1[:-1], ge_next_list[:-1], 'o', markersize=5, label=r'U + $\Delta$' )
#   ax.hlines(y=0, xmin=10., xmax=30, color='r', linewidth=0.5)
#   ax.set_xlim(x_min, x_max)
#   ax.set_ylim(y_min, y_max)
#   ax2 = ax.twinx()
#   ax2.plot(x_0, np.log10(np.array(temp_list)), label='Temp')
#   ax2.set_ylabel(r"Log Temperature [$K$]")
#   ax.set_ylabel(r"Internal Energy")
#   ax.legend(loc=0, fontsize=6)
#   ax2.legend(loc=4, fontsize=6)
# 
# 
# 
# fileName = 'error_cells_1.png'
# fig.tight_layout()
# fig.savefig( outDir + fileName )
# print 'Saved image: ', fileName
# print ''






