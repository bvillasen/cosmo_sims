import os, sys
import numpy as np
import h5py as h5

cosmo_dir = '/home/bruno/Desktop/Dropbox/Developer/cosmo_sims/'
toolsDirectory = cosmo_dir + "tools/"
sys.path.append( toolsDirectory )
from halo_finder import find_halos
from plot_halos import plot_halos_positions


dataDir = '/home/bruno/Desktop/data/'
inDir = dataDir +  'cosmo_sims/cholla_pm/cosmo_256_dm/'
outDir = inDir


Lbox =  115e3
n_slice = 64
snap = 19
# for snap in range(19, 20):
print "n_snap: ", snap
inFileName = 'data/particles_{0}.h5'.format(snap)
inFile = h5.File( inDir + inFileName )
dens = inFile['density'][...]
inFile.close()

dens = dens[:n_slice,:,:]

halo_limit = dens.mean() * 200
dens_limit = dens.mean() * 1
halo_catalog = find_halos( Lbox, dens, dens_limit, halo_limit )

# h_mass = halo_catalog[:,2]
#
# m_min = 1e13
# indxs = h_mass > m_min
# h_mass = h_mass[indxs]

plot_halos_positions( Lbox, halo_catalog, dens,  outDir, snap )
