import os, sys
import numpy as np
import h5py as h5
import matplotlib.pyplot as plt

cosmo_dir = '/home/bruno/Desktop/Dropbox/Developer/cosmo_sims/'
toolsDirectory = cosmo_dir + "tools/"
sys.path.append( toolsDirectory )
from halo_finder import find_halos
from plot_halos import plot_halos_positions
from load_halo_catalogs import load_asciiFiles
from mass_function import get_mass_function


dataDir = '/home/bruno/Desktop/data/'
inDir = dataDir +  'cosmo_sims/cholla_pm/cosmo_256_dm/'
halos_dir = inDir + 'halos/'
outDir = inDir

snapshots = [19]
# Load Rockstar Catalogs
rks_data_all = load_asciiFiles( snapshots, 8, halos_dir )

Lbox =  115e3
n_slice = 64
nSnap = 19

file_name = 'mass_function_{0}.png'.format(nSnap)
cumulative = True

fig = plt.figure(0)
plt.clf()
ax = plt.gca()

# for nSnap in range(19, 20):
print "n_nSnap: ", nSnap
inFileName = 'data/particles_{0}.h5'.format(nSnap)
inFile = h5.File( inDir + inFileName )
dens = inFile['density'][...]
inFile.close()

dens = dens[:n_slice,:,:]

halo_limit = dens.mean() * 172
dens_limit = dens.mean() * 1
halo_catalog = find_halos( Lbox, dens, dens_limit, halo_limit )
h_mass = halo_catalog[:,2]
# pos_x = halo_catalog[:,4]
# pos_y = halo_catalog[:,5]
# pos_z = halo_catalog[:,6]
# plot_halos_positions( Lbox, halo_catalog, dens,  outDir, nSnap )
# binCenters, massFunction = get_mass_function(h_mass, 1, comulative=cumulative, nBins=40)
# ax.plot( binCenters, massFunction )

rks_data = rks_data_all[nSnap]
h_mass = rks_data['mvir']
binCenters, massFunction = get_mass_function(h_mass, 1, comulative=cumulative, nBins=40)
ax.plot( binCenters, massFunction, label='Rockstar')


ax.set_xlabel(r'Mass $[{\rm h^{-1} M_{\odot}}]$', fontsize=15 )
if cumulative: ax.set_ylabel(r'n(>M)  $[h^3{\rm Mpc^{-3}}]$', fontsize=15 )
else: ax.set_ylabel(r'n $[h^3{\rm Mpc^{-3}}]$', fontsize=15 )
ax.legend( loc=1, prop={'size':13} )
ax.set_title('Cumulative Halo Mass function z=0', fontsize=15)
ax.set_xscale('log')
ax.set_yscale('log')
fig.savefig(outDir + file_name, bbox_inches='tight')
