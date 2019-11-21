import os, sys
from os import listdir
from os.path import isfile, join
from data_compress_grid import compress_grid
from data_compress_particles import compress_particles
from tools import create_directory
import numpy as np

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nSnap = rank

# rank = 0

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


if rank == 0:
  print 'Input Dir: ', inDir
  print 'Output Dir: ', outDir
  create_directory( outDir )



def split_name( file_name, part=False):
  nSnapshot, name, nBox = file_name.split('.')
  if part:
    indx = nSnapshot.find("_particles")
    nSnapshot = nSnapshot[:indx]
  return [int(nSnapshot), int(nBox)]


name_base = 'h5'

fields_hydro = ['density', 'temperature', 'HI_density', 'HII_density']
fields_hydro = 'all'

fields_particles = ['density', 'grav_potential']



if hydro:
  dataFiles = [f for f in listdir(inDir) if (isfile(join(inDir, f)) and (f.find('.h5.') > 0 ) and ( f.find('_particles') < 0) ) ]
else:
  dataFiles = [f for f in listdir(inDir) if (isfile(join(inDir, f)) and ( f.find('_particles') > 0) ) ]
dataFiles = np.sort( dataFiles )
nFiles = len( dataFiles )

files_names = np.array([ split_name( file_name, part=particles ) for file_name in dataFiles ])
snaps, boxes = files_names.T
snaps = np.unique( snaps )
boxes = np.unique( boxes )
snaps.sort()
nSnapshots = len( snaps )
nBoxes = len( boxes )

if rank == 0:
  print "Number of boxes: {0}".format(nBoxes)
  print "Number of snapshots: {0}".format(nSnapshots)

comm.Barrier()

if rank < nSnapshots:
  nSnap = rank
# for nSnap in range( nSnapshots):
  if hydro:
    out_base_name = 'grid_'
    compress_grid( nSnap, nBoxes, name_base, out_base_name, inDir, outDir, fields=fields_hydro )
  if cosmo or particles:
    out_base_name = 'particles_'
    compress_particles( nSnap, nBoxes, name_base, out_base_name, inDir, outDir , cosmology=cosmo, fields=fields_particles )
