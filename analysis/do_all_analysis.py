import sys, os
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import subprocess


currentDirectory = os.getcwd()
cosmo_dir = '/home/bruno/Desktop/Dropbox/Developer/cosmo_sims/'

eta_1 = 0.001
eta_2 = 0.020
# beta_0 = 0.25
# beta_1 = 0.00

params = ' {0:.3f} {1:.3f} '.format( eta_1, eta_2 )

print params
# 
# cmd = 'mpirun -n 28 python thermal_state_cool.py' + params
# subprocess.call([ cmd ], shell=True )
# # 
cmd = 'mpirun -n 28 python cooling/plot_chemistry.py' + params
subprocess.call([ cmd ], shell=True )
# 
# cmd = 'mpirun -n 10 python plot_power_spectrum_ch_cool_MPI.py' + params
# subprocess.call([ cmd ], shell=True )


