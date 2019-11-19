import sys
import numpy as np
# import matplotlib.pyplot as plt
import h5py as h5




def load_uv_file( fileName ):
  uvb_dir = {}

  file = h5.File( fileName, 'r' )
  cooling_rates = file['CoolingRates']
  uvb_rates = file['UVBRates']

  uvb_info = uvb_rates['Info'][...]
  uvb_z = uvb_rates['z'][...]
  uvb_dir['info'] = uvb_info
  uvb_dir['z'] = uvb_z

  uvb_chemistry = {} 
  for key in uvb_rates['Chemistry'].keys():
    uvb_chemistry[key] = uvb_rates['Chemistry'][key][...] 
  
  uvb_photoheating = {} 
  for key in uvb_rates['Photoheating'].keys():
    uvb_photoheating[key] = uvb_rates['Photoheating'][key][...] 

  uvb_dir['chemisty'] = uvb_chemistry
  uvb_dir['photoheating'] = uvb_photoheating
  return uvb_dir
  
  

def load_cooling_rates( fileName ):
  file = h5.File( fileName, 'r' )
  cooling_rates = {}
  for key in file['CoolingRates'].keys():
    group = file['CoolingRates'][key]
    cooling_rates[key] = {}
    for type in group.keys():
      cooling_rates[key][type] = group[type][...] 
  return cooling_rates


# 
# fileName = 'CloudyData_UVB=HM2012.h5'
# uvb_rates_HM = load_uv_file( fileName )
# cooling_rates_HM = load_cooling_rates( fileName )
# file = h5.File( fileName, 'r' )
# 
# 
# fileName = 'CloudyData_UVB=FG2011.h5'
# uvb_rates_FG = load_uv_file( fileName )
# cooling_rates_FG = load_cooling_rates( fileName )
# 
# 
# fileName = 'CloudyData_noUVB.h5'
# cooling_rates_noUVB = load_cooling_rates( fileName )
# 
# for key in cooling_rates_HM.keys():
#   group = cooling_rates_HM[key]
#   for type in group.keys():
#     vals_HM = cooling_rates_HM[key][type]  
#     vals_FG = cooling_rates_FG[key][type]  
#     vals_noUVB = cooling_rates_noUVB[key][type]
#     diff = np.max( np.abs(vals_noUVB - vals_HM) )
#     print diff
# 
# 






