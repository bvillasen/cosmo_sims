import sys
import numpy as np
# import matplotlib.pyplot as plt
import h5py as h5
from load_uv_file import load_uv_file, load_cooling_rates



fileName = 'CloudyData_noUVB.h5'
file = h5.File( fileName, 'r' )


rates = file['CoolingRates']
rates_primordial = rates['Primordial']

rates_primordial_cooling = rates_primordial['Cooling'][...]


rates_metals = rates['Metals']