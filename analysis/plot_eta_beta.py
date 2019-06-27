import sys
import numpy as np
import matplotlib.pyplot as plt

dev_dir = '/home/bruno/Desktop/Dropbox/Developer/'
cosmo_dir = dev_dir + 'cosmo_sims/'

inDir = '/home/bruno/cholla/'
outDir = dev_dir + 'figures/'

file_name = 'dual_energy_eta_beta.dat'
data = np.loadtxt( inDir + file_name )

# eta, beta = data.T
# log_eta = np.log10( eta )



eta_0 = 0.001
eta_1 = 1.0
beta_0 = 0.5
beta_1 = 0.1

nPoints = 100

y = np.linspace( beta_0, beta_1, nPoints)
x = np.linspace( np.log10(eta_0), np.log(eta_1), nPoints)


out_file_name = 'eta_beta.png'
plt.figure(0)
plt.clf()
plt.plot( x, y, linewidth=3 )
plt.plot( [-4, -3], [beta_0, beta_0], linestyle='--', c='C1', linewidth=1 )
plt.plot( [-4, 0], [beta_1, beta_1], linestyle='--', c='C1', linewidth=1 )
plt.axvline( np.log10(eta_0), ymin=beta_0, c='C0', linewidth=3 )
plt.axvline( np.log10(eta_1), ymin=beta_1, c='C0', linewidth=3 )
plt.axvline( np.log10(eta_0), ymin=0, ymax=beta_0, linestyle='--', c='C2', linewidth=1 )
# plt.axvline( np.log10(eta_1), ymin=0, ymax=beta_1, linestyle='--', c='C1', linewidth=1 )
plt.text( -3.8, 0.52, r'$\beta_0 = 0.5$ ', fontsize=10)
plt.text( -0.7, 0.05, r'$\beta_1 = 0.1$ ', fontsize=10)
plt.text(  -2.95, 0.35, r'$\eta_0 = 0.001$ ', fontsize=10, rotation='vertical')
plt.text( -2.3, 0.8, r'Use Internal Energy ', fontsize=11)
plt.text( -2.2, 0.75, r'from Total Energy ', fontsize=11)
plt.title("Condition for Dual Energy")
# plt.xscale('log')
plt.xlim(-4, 1 )
plt.ylim(0, 1 )
plt.xlabel( r'$ log( \eta )$' )
plt.ylabel( r'$ \beta $' )


plt.savefig( outDir + out_file_name,  bbox_inches='tight', dpi=200)