import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
from load_uv_file import load_uv_file, load_cooling_rates

fileName = 'CloudyData_UVB=HM2012.h5'
file_uv = h5.File( fileName, 'r' )



fileName = 'CloudyData_noUVB.h5'
file = h5.File( fileName, 'r' )

# 
# fileName = 'CloudyData_UVB=HM2012.h5'
# file = h5.File( fileName, 'r' )
# uvb_rates_HM = load_uv_file( fileName )
# cooling_rates_HM = load_cooling_rates( fileName )
# z = uvb_rates_HM['z']
# uvb_HI = uvb_rates_HM['photoheating']['piHI']
# uvb_HeI = uvb_rates_HM['photoheating']['piHeI']
# uvb_HeII = uvb_rates_HM['photoheating']['piHeII']
# 
# 
# #Plot UVB uvb_rates
# nrows=1
# ncols = 1
# fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10*ncols,8*nrows))
# 
# lw = 3
# ax.plot( z, uvb_HI, linewidth=lw, label="HI" )
# ax.plot( z, uvb_HeI, linewidth=lw, label="HeI" )
# ax.plot( z, uvb_HeII, linewidth=lw, label="HeII" )
# 
# ax.set_yscale('log')
# ax.legend()
# ax.set_title('UVB Rates H&M 2012')
# ax.set_xlabel('Redshift')
# ax.set_ylabel('Photo-Heating Rate')
# 
# fig.savefig( 'uvb_rates.png',  bbox_inches='tight', dpi=100)





