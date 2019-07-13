import sys, time, os
import numpy as np
import matplotlib.pyplot as plt
from subprocess import call
from shutil import copyfile


dev_dir = '/home/bruno/Desktop/Dropbox/Developer/'
# inDir = cosmo_dir + 'figures/cosmo_256_cholla_highRes/'
# inDir = cosmo_dir + 'figures/collapse/anim/'
# inDir = dev_dir + 'figures/chemistry/chemistry_HI_eta0.005_beta0.250_0.000_PressureJump10.0/'
inDir = dev_dir + 'figures/phase_diagram/uvb_eta0.001_beta0.000_0.000_noDE/'
# inDir = cosmo_dir + 'figures/projections/de1_PPMC_HLLC/'

# inDir = cosmo_dir + 'figures/zeldovich/enzo_simple_beta_convDE/'
outDir = '/home/bruno/Desktop/'

# image_name = 'chemistry'
image_name = 'phase_diagram'
out_anim_name = 'phase_diagram_DE_beta_simple_beta0.00_noDE'
# image_name = 'zeldovich'
# out_anim_name = 'zeldovich_enzo_simple_beta_convDE'

cmd = 'ffmpeg -framerate 1  '
# cmd += ' -start_number 45'
cmd += ' -i {0}{1}_%d.png '.format( inDir, image_name )
cmd += '-pix_fmt yuv420p '
# cmd += '-b 50000k '
cmd += '{0}{1}.mp4'.format( outDir, out_anim_name )
cmd += ' -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2"'
os.system( cmd )


# n_files = 4
#
# n_repeat = 15
# for counter in range(1,n_repeat+1):
#   index_add = counter * n_files
#   for i in range( n_files ):
#     index_new = i + index_add
#     # print i, index_new
#     src = image_name + '_{0}.png'.format( i )
#     dst = image_name + '_{0}.png'.format( index_new )
#     print src, dst
#     copyfile(inDir + src, inDir + dst)


# inDir = '/home/bruno/Desktop/anim/'
# outDir = '/home/bruno/Desktop/anim/'
# image_name = 'gpu_model'
#
#
# out_anim_name = 'gpu_model'
