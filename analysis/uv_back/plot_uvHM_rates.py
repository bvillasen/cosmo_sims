import sys
import numpy as np
# import matplotlib.pyplot as plt
import h5py as h5
from load_uv_file import load_uv_file, load_cooling_rates



fileName = 'CloudyData_UVB=HM2012.h5'
file = h5.File( fileName, 'r' )


rates = file['CoolingRates']
rates_primordial = rates['Primordial']

#Get values for temperature in the table
temp_vals = rates_primordial['Cooling'].attrs['Temperature']
temp_vals_log = np.log10( temp_vals )

#Get values for density in the table
dens_vals = rates_primordial['Cooling'].attrs['Parameter1']

#Get values for redshitt in the table
redshift_vals = rates_primordial['Cooling'].attrs['Parameter2']