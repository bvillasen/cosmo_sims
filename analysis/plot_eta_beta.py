import sys
import numpy as np
import matplotlib.pyplot as plt

cosmo_dir = '/home/bruno/Desktop/Dropbox/Developer/cosmo_sims/'

inDir = '/home/bruno/cholla/'
outDir = cosmo_dir + 'figures/'

file_name = 'dual_energy_eta_beta.dat'
data = np.loadtxt( inDir + file_name )

eta, beta = data.T

log_eta = np.log10( eta )

out_file_name = 'eta_beta.png'
plt.figure(0)
plt.plot( log_eta, beta )
plt.xlim(-4, 1 )
plt.ylim(0, 1 )
plt.xlabel( r'$ log( \eta )$' )
plt.ylabel( r'$ \beta $' )


plt.savefig( outDir + out_file_name )