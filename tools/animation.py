import sys, time, os
import numpy as np
import matplotlib.pyplot as plt
from subprocess import call
from shutil import copyfile


dev_dir = '/home/bruno/Desktop/Dropbox/Developer/'
# inDir = cosmo_dir + 'figures/cosmo_256_cholla_highRes/'
# inDir = cosmo_dir + 'figures/collapse/anim/'
# inDir = dev_dir + 'figures/chemistry/chemistry_HI_eta0.001_0.030/'
# inDir = dev_dir + 'figures/phase_diagram/uvb_SIMPLE_several/'
# inDir = dev_dir + 'figures/power_hydro/animation/'
# inDir = dev_dir + 'figures/cell_difference/'
# inDir = dev_dir + 'figures/zeldovich/'
# inDir = dev_dir + 'figures/spectra/'
inDir = dev_dir + 'figures/dm_projection_50Mpc/'

# inDir = '/home/bruno/Desktop/namrata/'




outDir = '/home/bruno/Desktop/'

# image_name = 'l1_cell_difference'
# image_name = 'chemistry'
# image_name = 'phase_diagram'
# image_name = 'ps_128_cooling_uv_PPMC_HLLC_SIMPLE'
# out_anim_name = 'l1_difference_eta2'
# image_name = 'zeldovich'
# image_name = 'spectra'
image_name = 'projection'

# out_anim_name = 'zeldovich_error_noGrav'
out_anim_name = 'dm_projection_50Mpc'
# out_anim_name = 'spec_animation'

cmd = 'ffmpeg -framerate 10  '
# cmd += ' -start_number 45'
cmd += ' -i {0}{1}_%d.png '.format( inDir, image_name )
cmd += '-pix_fmt yuv420p '
# cmd += '-b 50000k '
cmd += '{0}{1}.mp4'.format( outDir, out_anim_name )
cmd += ' -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2"'
cmd += ' -vf pad="width=ceil(iw/2)*2:height=ceil(ih/2)*2"'
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
