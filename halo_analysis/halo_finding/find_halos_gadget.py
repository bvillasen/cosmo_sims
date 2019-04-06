import sys, os, time
from subprocess import call

from mpi4py import MPI
MPIcomm = MPI.COMM_WORLD
pId = MPIcomm.Get_rank()
nProc = MPIcomm.Get_size()

currentDirectory = os.getcwd()
#Add Modules from other directories
devDirectory = '/home/bruno/Desktop/Dropbox/Developer/'
toolsDirectory = devDirectory + "tools/"
sys.path.append( toolsDirectory )
# from tools import *
#import haloAnalysis as haloA

dataDir = '/raid/bruno/data/cosmo_sims/gadget/cosmo_1024/'

rockstarDir = devDirectory + 'cosmo_sims/halo_analysis/halo_finding/rockstar/'
rockstarComand = rockstarDir + 'rockstar'

h = 0.6774
rockstarConf = {
'FILE_FORMAT': '"GADGET2"',
'GADGET_LENGTH_CONVERSION' :1e-3,  #convert from kpc to Mpc
'GADGET_MASS_CONVERSION': 1e+10,
'FORCE_RES': 2.426e-3,                 #Mpc/h
'OUTBASE': dataDir + 'halo_catalogs/',
}
parallelConf = {
'PARALLEL_IO': 1,
'PERIODIC': 1,                                #non-periodic boundary conditions
'INBASE':  dataDir ,               #"/directory/where/files/are/located"
'NUM_BLOCKS': 8,                              # <number of files per snapshot>
'NUM_SNAPS': 6,                               # <total number of snapshots>
'STARTING_SNAP': 5,
'FILENAME': '"snapshot_<snap>.<block>"',              #"my_sim.<snap>.<block>"
# 'SNAPSHOT_NAMES': dataDir + 'halos/snaps_names.txt',
# 'BGC2_SNAPNAMES': dataDir + 'halos/snaps_names.txt',
'NUM_WRITERS': 8,                             #<number of CPUs>
'FORK_READERS_FROM_WRITERS': 1,
'FORK_PROCESSORS_PER_MACHINE': 8,             #<number of processors per node>
}

if pId == 0:
  if not os.path.exists( rockstarConf['OUTBASE']): os.makedirs(rockstarConf['OUTBASE'])
  rockstarconfigFile = rockstarConf['OUTBASE'] + '/rockstar_param.cfg'
  rckFile = open( rockstarconfigFile, "w" )
  for key in rockstarConf.keys():
    rckFile.write( key + " = " + str(rockstarConf[key]) + "\n" )
  for key in parallelConf.keys():
    rckFile.write( key + " = " + str(parallelConf[key]) + "\n" )
  rckFile.close()
  #Run ROCKSTAR finder
  print "\nFinding halos..."
  print " Parallel configuration"
  print "Output: ", rockstarConf['OUTBASE'] + '\n'

MPIcomm.Barrier()
start = time.time()
if pId == 0: call([rockstarComand, "-c", rockstarconfigFile ])
if pId == 1:
  time.sleep(5)
  call([rockstarComand, "-c", rockstarConf['OUTBASE'] + '/auto-rockstar.cfg' ])
print "Time: ", time.time() - start
