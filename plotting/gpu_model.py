import numpy as np
import matplotlib.pyplot as plt


n_lines = 12
n_blocks = 12
n_sub_block = 3
n_per_block = n_blocks / n_sub_block

L = 1.
dx = L/n_blocks
data = np.zeros( (n_blocks, n_blocks) )
x_lines = np.linspace(0,L,n_blocks+1)

c_map = 'Paired'
c_map = 'tab10'
outDir = '/home/bruno/Desktop/gpu_anim/'

fig, ax = plt.subplots(nrows=1, ncols=1)
fig.set_size_inches(10,10)
plt.tight_layout()

n_image = 4
counter = 2
data_1 = np.zeros( (n_blocks, n_blocks) )
 
i_block = 0
j_block = 0

for i in range(n_blocks):
  for j in range(n_blocks):
    if ( i>=i_block*n_per_block and i<(i_block+1)*n_per_block  and j>=j_block*n_per_block and j<(j_block+1)*n_per_block):
      data[i][j] = counter
      counter += 1
      if counter >= 11 : counter =2
# 
# for i in range(n_blocks):
#   for j in range(n_blocks):
#     if ( i<n_blocks/2 and j>=n_blocks/2):
#       data[i][j] = counter
#       counter += 1
#       if counter >= 11 : counter =1
# 
# for i in range(n_blocks):
#   for j in range(n_blocks):
#     if ( i>=n_blocks/2 and j<n_blocks/2):
#       data[i][j] = counter
#       counter += 1
#       if counter >= 11 : counter =1
# 
# for i in range(n_blocks):
#   for j in range(n_blocks):
#     if ( i>=n_blocks/2 and j>=n_blocks/2):
#       data[i][j] = counter
#       counter += 1
#       if counter >= 11 : counter =1


ax.imshow(data, extent=(0,1,0,1), cmap=c_map, vmin=1, vmax=10)
ax.vlines(x_lines, ymin=0, ymax=1, colors='w')
ax.hlines(x_lines, xmin=0, xmax=1, colors='w')


ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_facecolor('xkcd:black')
ax.yaxis.set_visible('false')
ax.xaxis.set_visible('false')
ax.set_xticklabels([])
ax.set_yticklabels([])
fig.savefig( outDir + 'gpu_model_{0}.png'.format(n_image), pad_inches=0,  bbox_inches='tight')
n_image += 1

# 
# from shutil import copyfile
# offset= 5
# n = 0
# for n in range(1,10):
#   for i in range(0,5):
#     src = outDir + 'gpu_model_{0}.png'.format(i)
#     dst = outDir + 'gpu_model_{0}.png'.format(i + n*offset )
#     copyfile(src, dst)
