import os, sys
from os import listdir
from os.path import isfile, join
import h5py
import numpy as np
currentDirectory = os.getcwd()
#Add Modules from other directories
toolsDirectory = currentDirectory
sys.path.append( toolsDirectory )
from tools import *
import glio




dataDir = '/home/bruno/Desktop/hard_drive_1/data/'
gadgetDir = dataDir + 'pleiades/gadget/1024_dm/data/'


nSnap = 54
nBox = 1
  # for nBox in range(nBoxes):
snapKey = '_{0:03}.{1}'.format( nSnap, nBox)
inFileName = 'snapshot{0}'.format( snapKey)
print '\nLoading Gadget file:', inFileName
s = glio.GadgetSnapshot( gadgetDir + inFileName )
s.load()
head = s.header
fields = s.fields