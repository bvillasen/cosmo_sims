import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5

dev_dir = '/home/bruno/Desktop/Dropbox/Developer/'
cosmo_dir = dev_dir + 'cosmo_sims/'
toolsDirectory = cosmo_dir + "tools/"
sys.path.extend([toolsDirectory ] )
from load_data_cholla import load_snapshot_data
from load_data_enzo import load_snapshot_enzo
from phase_diagram import get_phase_diagram
from internal_energy import get_internal_energy, get_temp, get_Temperaure_From_Flags_DE
from tools import create_directory


dataDir = '/raid/bruno/data/'

outDir = dev_dir + 'figures/dual_energy_enzo/'
create_directory( outDir )
print "Output: ", outDir

Lbox = 50000.
nPoints = 128 


enzoDir = dataDir + 'cosmo_sims/enzo/{0}_cool_uv/h5_files/'.format(nPoints)
enzoDir_noDE = dataDir + 'cosmo_sims/enzo/{0}_cool_uv_noDE/h5_files/'.format(nPoints)



def get_dual_energy_indxs( gas_energy, total_energy, eta ):
  indxs = np.where( gas_energy / total_energy < eta )
  return indxs


def get_dual_energy_indxs_2( gas_energy, total_energy, eta_2 ):
  nz, ny, nx = gas_energy.shape
  ids_k, ids_j, ids_i = [], [], []
  for k in range(nz):
    k_l = k-1 if k > 0 else nz-1 
    k_r = k+1 if k < nz-1 else 0 
    for j in range(ny):
      j_l = j-1 if j > 0 else ny-1 
      j_r = j+1 if j < ny-1 else 0 
      for i in range(nx):
        i_l = i-1 if i > 0 else nx-1 
        i_r = i+1 if i < nx-1 else 0
        
        E_c = total_energy[k, j, i]
        E_r = total_energy[k, j, i_r]
        E_l = total_energy[k, j, i_l]
        E_u = total_energy[k, j_r, i]
        E_d = total_energy[k, j_l, i]
        E_t = total_energy[k_r, j, i]
        E_b = total_energy[k_l, j, i]
        
        E_max = max([E_c, E_r, E_l, E_u, E_d, E_t, E_b ])
        
        U = gas_energy[k, j, i]
        if ( U / E_max < eta_2 ):
          ids_k.append(k)
          ids_j.append(j)
          ids_i.append(i)
        
  ids_k = np.array(ids_k)
  ids_j = np.array(ids_j)
  ids_i = np.array(ids_i)
  return ( ids_k, ids_j, ids_i )
      
      
      

  

gamma = 5./3
nx = nPoints
ny = nPoints
nz = nPoints
ncells = nx * ny * nz

dv = (Lbox/nPoints)**3
nbins = 1000




nSnap = 6
# for nSnap in range(12):
fileName = 'dual_energy_enzo_{0}_condition_2.png'.format( nSnap )


data_enzo = load_snapshot_enzo( nSnap, enzoDir, dm=False, cool=True)
current_z_en_DE = data_enzo['current_z']
dens_en = data_enzo['gas']['density'][...]
GasEnergy = data_enzo['gas']['GasEnergy'][...] / dens_en
temp_en_0 =  data_enzo['gas']['temperature'][...]
dens_mean = dens_en.mean()
rho_en_DE = dens_en.reshape( nx*ny*nz ) / dens_mean
temp_en_DE = temp_en_0.reshape( nx*ny*nz )
x_en_DE, y_en_DE, z_en_DE = get_phase_diagram( rho_en_DE, temp_en_DE , nbins, ncells )


data_enzo = load_snapshot_enzo( nSnap, enzoDir_noDE, dm=False, cool=True)
current_z_en = data_enzo['current_z']
dens_en = data_enzo['gas']['density'][...]
GasEnergy = data_enzo['gas']['GasEnergy'][...] / dens_en
vx = data_enzo['gas']['momentum_x'][...] / dens_en
vy = data_enzo['gas']['momentum_y'][...] / dens_en
vz = data_enzo['gas']['momentum_z'][...] / dens_en
temp_en =  data_enzo['gas']['temperature'][...]
temp_DE_1 = temp_en.copy()
temp_DE_2 = temp_en.copy()
temp_DE_3 = temp_en.copy()
v2 = vx*vx + vy*vy + vz*vz
Ekin = 0.5 * v2
TotaEnrgy = GasEnergy + Ekin
dens_mean = dens_en.mean()
rho_en = dens_en.reshape( nx*ny*nz ) / dens_mean
temp_en = temp_en.reshape( nx*ny*nz )


x_en, y_en, z_en = get_phase_diagram( rho_en, temp_en , nbins, ncells )


indxs_DE_local_E = get_dual_energy_indxs_2( GasEnergy * dens_en, TotaEnrgy* dens_en , 0.1 )

eta_1 = 0.001
indxs_DE = get_dual_energy_indxs( GasEnergy, TotaEnrgy, eta_1 )
temp_DE_1[indxs_DE] = temp_en_0[indxs_DE]  
temp_DE_1[indxs_DE_local_E] = temp_en_0[indxs_DE_local_E]  
temp_en_DE_1 = temp_DE_1.reshape( nx*ny*nz )
x_en_1, y_en_1, z_en_1 = get_phase_diagram( rho_en, temp_en_DE_1 , nbins, ncells )  




eta_2 = 0.1
indxs_DE = get_dual_energy_indxs( GasEnergy, TotaEnrgy, eta_2 )
temp_DE_2[indxs_DE] = temp_en_0[indxs_DE]  
temp_DE_2[indxs_DE_local_E] = temp_en_0[indxs_DE_local_E]  
temp_en_DE_2 = temp_DE_2.reshape( nx*ny*nz )
x_en_2, y_en_2, z_en_2 = get_phase_diagram( rho_en, temp_en_DE_2 , nbins, ncells )  

eta_3 = 0.6
indxs_DE = get_dual_energy_indxs( GasEnergy, TotaEnrgy, eta_3 )
temp_DE_3[indxs_DE] = temp_en_0[indxs_DE]  
temp_DE_3[indxs_DE_local_E] = temp_en_0[indxs_DE_local_E]  
temp_en_DE_3 = temp_DE_3.reshape( nx*ny*nz )
x_en_3, y_en_3, z_en_3 = get_phase_diagram( rho_en, temp_en_DE_3 , nbins, ncells )  



nrows = 1
ncols = 5
fig, ax_l = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10*ncols,8*nrows))
fig.clf()

x_min = -2
x_max = 2
x_min_h = -11
x_max_h = 3

y_max = 5




n = 0
plt.subplot(nrows, ncols, n*ncols+1)
ax = plt.gca()
ax.clear()
c = ax.scatter( y_en_DE, x_en_DE, c = np.log10(z_en_DE), s=1  )
plt.colorbar(c)
ax.set_ylabel(r'Log Temperature $[K]$', fontsize=15 )
ax.set_xlabel(r'Log Gas Overdensity', fontsize=15 )
ax.set_title( " Z={0:.2f}  Dual Energy ON ".format( current_z_en_DE),fontsize=17)
ax.set_xlim(x_min, x_max)
ax.set_ylim(-1, y_max)
# 

plt.subplot(nrows, ncols, n*ncols+2)
ax = plt.gca()
ax.clear()
c = ax.scatter( y_en, x_en, c = np.log10(z_en), s=1  )
plt.colorbar(c)
ax.set_ylabel(r'Log Temperature $[K]$', fontsize=15 )
ax.set_xlabel(r'Log Gas Overdensity', fontsize=15 )
ax.set_title( " Z={0:.2f}  Dual Energy OFF ".format( current_z_en),fontsize=17)
ax.set_xlim(x_min, x_max)
ax.set_ylim(-1, y_max)


plt.subplot(nrows, ncols, n*ncols+3)
ax = plt.gca()
ax.clear()
c = ax.scatter( y_en_1, x_en_1, c = np.log10(z_en_1), s=1  )
plt.colorbar(c)
ax.set_ylabel(r'Log Temperature $[K]$', fontsize=15 )
ax.set_xlabel(r'Log Gas Overdensity', fontsize=15 )
ax.set_title( " Z={0:.2f}  Dual Energy Condition ".format( current_z_en),fontsize=17)
ax.set_xlim(x_min, x_max)
ax.set_ylim(-1, y_max)
ax.text(1, 0.9*y_max, r'$\eta_1 = {0:.3f}$'.format(eta_1), fontsize=17  )
ax.text(1, 0.8*y_max, r'$\eta_2 = {0:.3f}$'.format(eta_2), fontsize=17  )



plt.subplot(nrows, ncols, n*ncols+4)
ax = plt.gca()
ax.clear()
c = ax.scatter( y_en_2, x_en_2, c = np.log10(z_en_2), s=1  )
plt.colorbar(c)
ax.set_ylabel(r'Log Temperature $[K]$', fontsize=15 )
ax.set_xlabel(r'Log Gas Overdensity', fontsize=15 )
ax.set_title( " Z={0:.2f}  Dual Energy Condition ".format( current_z_en),fontsize=17)
ax.set_xlim(x_min, x_max)
ax.set_ylim(-1, y_max)
ax.text(1, 0.9*y_max, r'$\eta_1 = {0:.3f}$'.format(eta_2), fontsize=17  )
ax.text(1, 0.8*y_max, r'$\eta_2 = {0:.3f}$'.format(eta_2), fontsize=17  )


plt.subplot(nrows, ncols, n*ncols+5)
ax = plt.gca()
ax.clear()
c = ax.scatter( y_en_3, x_en_3, c = np.log10(z_en_3), s=1  )
plt.colorbar(c)
ax.set_ylabel(r'Log Temperature $[K]$', fontsize=15 )
ax.set_xlabel(r'Log Gas Overdensity', fontsize=15 )
ax.set_title( " Z={0:.2f}  Dual Energy Condition ".format( current_z_en),fontsize=17)
ax.set_xlim(x_min, x_max)
ax.set_ylim(-1, y_max)
ax.text(1, 0.9*y_max, r'$\eta_1 = {0:.3f}$'.format(eta_3), fontsize=17  )
ax.text(1, 0.8*y_max, r'$\eta_2 = {0:.3f}$'.format(eta_2), fontsize=17  )


fig.tight_layout()
fig.savefig( outDir + fileName )
print 'Saved image: ', fileName
print ''
  #
