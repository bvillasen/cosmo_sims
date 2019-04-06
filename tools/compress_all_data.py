import sys
from compress_data_grid import compress_grid
from compress_data_particles import compress_particles
from tools import create_directory


n_param = len( sys.argv )
dataDir = '/raid/bruno/data/'
inDir = dataDir + 'cosmo_sims/cholla_pm/256_cool/'
outDir = inDir + 'data/'
cosmo = True
hydro = True
particles = False
if n_param == 2: outDir = inDir + sys.argv[1]
if n_param >= 3:
  inDir, outDir = sys.argv[1], sys.argv[2]
  outDir = inDir + outDir
if inDir[-1] != "/": inDir += '/'
if outDir[-1] != "/": outDir += '/'
for option in sys.argv:
  if option == 'no_cosmo': cosmo = False
  if option == 'particles': particles, hydro = True, False

print 'Input Dir: ', inDir
print 'Output Dir: ', outDir
create_directory( outDir )
if hydro: compress_grid( inDir, outDir )
if cosmo or particles: compress_particles( inDir, outDir, cosmology=cosmo )
