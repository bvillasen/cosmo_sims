import os, sys
from os import listdir
from os.path import isfile, join
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import root, newton, fsolve
import socket
# # from sympy import *
cosmo_dir = '/home/bruno/Desktop/Dropbox/Developer/cosmo_sims/'
toolsDirectory = cosmo_dir + "tools/"
sys.path.extend([toolsDirectory ] )

dataDir = '/raid/bruno/data/'

inputDir_cufft = dataDir + 'cosmo_sims/cholla_pm/sphere_collapse/'
inputDir_pfft = dataDir + 'cosmo_sims/cholla_pm/sphere_collapse/data_pfft_256/'

nSnap = 0;

for nSnap in range(11):
  file_name = '{0}.h5'.format(nSnap)
  file = h5.File( inputDir_cufft + file_name, 'r' )
  dens_cufft = file['density'][...] 
  # pot_cufft = file['potential'][...]
  file.close()

  file_name = 'grid_{0}.h5'.format(nSnap)
  file = h5.File( inputDir_pfft + file_name, 'r' )
  dens_pfft = file['density'][...] 
  # pot_pfft = file['potential'][...]
  file.close()

  diff = (dens_cufft - dens_pfft ) / dens_pfft
  print diff.min(), diff.max()
