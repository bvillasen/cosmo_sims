import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import yt

# dataDir = '/raid/bruno/data/cosmo_sims/'
dataDir = '/home/bruno/Desktop/data/'
cosmo_dir = '/home/bruno/Desktop/Dropbox/Developer/cosmo_sims/'
toolsDirectory = cosmo_dir + "tools/"
sys.path.extend([toolsDirectory ] )



dataDir = '/raid/bruno/data/'
enzoDir = dataDir + 'cosmo_sims/enzo/ZeldovichPancake_HLLC/'
outputDir = enzoDir + 'ics/'

print 'Output Dir: ', outputDir



L = 64.
n = 256
dx = L / ( n )
x = np.arange(0, 256, 1)* dx + 0.5*dx

nSnap = 27
# for nSnap in range( 80 ):

out_file_name = 'ics_zeldovich_{0}.dat'.format(nSnap)

file_name = enzoDir + 'DD{0:04}/data{0:04}'.format(nSnap)
ds = yt.load( file_name )
data = ds.all_data()
h = ds.hubble_constant
current_z = ds.current_redshift
current_a = 1./(current_z + 1)
print current_z
x = data[('gas', 'x')].in_units('Mpc/h').v / current_a
gas_dens = data[ ('gas', 'density')].in_units('msun/kpc**3').v*current_a**3/h**2
gas_temp = data[ ('gas', 'temperature')].v
gas_vel = data[ ('gas', 'velocity_x')].in_units('km/s').v
gas_u = data[('gas', 'thermal_energy' )].v * 1e-10 *gas_dens #km^2/s^2
# temp = get_temp(gas_u / gas_dens * 1e6, mu=mu)
Ekin = 0.5 * gas_dens * gas_vel * gas_vel
gas_E = Ekin + gas_u


ics_data = [ gas_dens, gas_vel, gas_E, gas_u ]
ics_data = np.array( ics_data).flatten()

np.savetxt( outputDir + out_file_name, ics_data )

