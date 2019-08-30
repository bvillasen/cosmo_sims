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

# dataDir = '/raid/bruno/data/'
dataDir = '/home/bruno/Desktop/data/'

inputDir_sor = dataDir + 'cosmo_sims/cholla_pm/sphere_collapse/'
inputDir_cufft = dataDir + 'cosmo_sims/cholla_pm/sphere_collapse/data_cufft/'

nSnap = 0;

for nSnap in range(11):
  file_name = '{0}.h5'.format(nSnap)
  file = h5.File( inputDir_sor + file_name, 'r' )
  dens_sor = file['density'][...] 
  pot_sor = file['potential'][...]
  file.close()

  file_name = '{0}.h5'.format(nSnap)
  file = h5.File( inputDir_cufft + file_name, 'r' )
  dens_cufft = file['density'][...] 
  pot_cufft = file['potential'][...]
  file.close()
  # 
  diff = (dens_sor - dens_cufft ) / dens_cufft
  print diff.min(), diff.max()
