import sys, os
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import subprocess


currentDirectory = os.getcwd()
cosmo_dir = '/home/bruno/Desktop/Dropbox/Developer/cosmo_sims/'

eta_0 = 0.005
beta_0 = 0.055
beta_1 = 0.000

params = ' {0:.3f} {1:.3f}  {2:.3f}'.format( eta_0, beta_0, beta_1 )

print params

cmd = 'mpirun -n 31 python thermal_state_cool_beta.py' + params
subprocess.call([ cmd ], shell=True )
# 
cmd = 'mpirun -n 31 python cooling/plot_chemistry.py' + params
subprocess.call([ cmd ], shell=True )

cmd = 'mpirun -n 10 python plot_power_spectrum_ch_256_cool_MPI.py' + params
subprocess.call([ cmd ], shell=True )


