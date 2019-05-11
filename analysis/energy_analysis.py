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
from load_data_enzo import load_snapshot_enzo
from tools import create_directory


dataDir = '/raid/bruno/data/'

enzoDir_uv = dataDir + 'cosmo_sims/enzo/256_cool_uv/h5_files/'


chollaDir_0 = dataDir + 'cosmo_sims/cholla_pm/256_cool/data_de001_PPMC_HLLC/'

outDir = cosmo_dir + 'figures/energy/'
create_directory( outDir )


fracc_list = []
z_list = []

nPoints = 256
n_cells = nPoints**3

percentile = 0.90
index = int( n_cells * percentile )

file_name = 'E_percentile_{0}.png'.format(int(percentile*100))

gamma = 5./3

nSnap = 0
for nSnap in range(31):
  data_enzo = load_snapshot_enzo( nSnap, enzoDir_uv, cool=True, metals=True )
  current_a_enzo = data_enzo['current_a']
  current_z_enzo = data_enzo['current_z']
  dens_en = data_enzo['gas']['density'][...]
  vel_x_en = data_enzo['gas']['momentum_x'][...] / dens_en
  vel_y_en = data_enzo['gas']['momentum_y'][...] / dens_en
  vel_z_en = data_enzo['gas']['momentum_z'][...] / dens_en
  E_en = data_enzo['gas']['Energy'][...]
  U_en = data_enzo['gas']['GasEnergy'][...]
  
  v2 = ( vel_x_en*vel_x_en + vel_y_en*vel_y_en + vel_z_en*vel_z_en  )
  Ekin_en = 0.5 * dens_en *  v2

  # c = np.sqrt(gamma  * U_en *(gamma - 1)/ dens_en)
  # c = c.flatten()
  # v = np.sqrt( v2.flatten() )
  # fracc = np.percentile( v/c, percentile ) 
   

  E_kin = Ekin_en.flatten()
  U = U_en.flatten()
  fracc = np.percentile( E_kin / U, percentile)


  
  
  fracc_list.append( fracc )
  z_list.append( current_z_enzo)




fig = plt.figure(0)
plt.clf()
ax = plt.gca()


ax.plot( z_list, fracc_list)


ax.set_xlabel(r'Redshift', fontsize=15 )
# ax.set_ylabel(r' $V / C_{sound} $', fontsize=15 )
# ax.set_title('{0} Percentile Velocity Fraction '.format(int(percentile*100)), fontsize=15)
ax.set_title('{0} Percentile Energy Fraction '.format(int(percentile*100)), fontsize=15)
ax.set_ylabel(r'$E_{kin} / U $', fontsize=15 )
# ax.set_xscale('log')
ax.set_yscale('log')
# ax.set_xlim(0, 100)
fig.savefig(outDir + file_name, bbox_inches='tight')

