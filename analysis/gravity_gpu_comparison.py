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
dev_dir = '/home/bruno/Desktop/Dropbox/Developer/'
cosmo_dir = dev_dir + 'cosmo_sims/'
toolsDirectory = cosmo_dir + "tools/"
sys.path.extend([toolsDirectory ] )
from load_data_cholla import load_snapshot_data

dataDir = '/raid/bruno/data/'

inputDir_cpu = dataDir + 'cosmo_sims/cholla_pm/sphere_collapse/data_cpu_simple/'
inputDir_gpu = dataDir + 'cosmo_sims/cholla_pm/sphere_collapse/data_gpu_simple_new/'

# nSnap = 0;
for nSnap in range(11):
  data_cholla = load_snapshot_data( nSnap, inputDir_cpu, cool=False, dm=False )
  dens_cpu = data_cholla['gas']['density'][...]

  data_cholla = load_snapshot_data( nSnap, inputDir_gpu, cool=False, dm=False )
  dens_gpu = data_cholla['gas']['density'][...]

  diff = ( dens_gpu - dens_cpu )/ dens_cpu

  print diff.min(), diff.max()

# 
# for nSnap in range(11):
#   file_name = '{0}.h5'.format(nSnap)
#   file = h5.File( inputDir_cufft + file_name, 'r' )
#   dens_cufft = file['density'][...] 
#   # pot_cufft = file['potential'][...]
#   file.close()
# 
#   file_name = 'grid_{0}.h5'.format(nSnap)
#   file = h5.File( inputDir_pfft + file_name, 'r' )
#   dens_pfft = file['density'][...] 
#   # pot_pfft = file['potential'][...]
#   file.close()
# 
#   diff = (dens_cufft - dens_pfft ) / dens_pfft
#   print diff.min(), diff.max()
