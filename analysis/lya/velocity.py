import sys
import numpy as np


current_z = 2.3
L = 115. #Mpc
n_points = 256
dz = L / n_points


H0 = 67.74                #[km/s / Mpc]
cosmo_h = H0/100
# H0 /= 1000                #[km/s / kpc]
Omega_M = 0.3089
Omega_L = 0.6911


current_a = 1. / ( current_z + 1 )
a_dot = np.sqrt( Omega_M/current_a + Omega_L*current_a**2  ) * H0 
H = a_dot / current_a
R = current_a * L / cosmo_h
dr = R / n_points

x_comov = np.linspace( 0, L, n_points )
r_proper = current_a * x_comov / cosmo_h
vel = H * r_proper   #km/sec

