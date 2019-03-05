import os, sys
import numpy as np


def get_mass_function(h_mass, boxSize, comulative=True, nBins=50, binEdges='None'):
  # nBins = 25
  masses = h_mass
  # masses_galaxies = np.array([ ms_halosData[snapshot][hId].attrs['SM'] for hId in galaxiesFilter ])
  if binEdges=='None':
    minMass, maxMass = masses.min(), masses.max()
    binEdges = np.exp(np.linspace(np.log(minMass), np.log(maxMass*(1.01)), nBins))
    binCenters = np.sqrt(binEdges[:-1]*binEdges[1:])
    hist = np.histogram(masses, bins=binEdges)[0]
  else:
    print ' Using binEdges for histogram'
    hist = np.histogram(masses, bins=binEdges)[0]
    binCenters = np.sqrt(binEdges[:-1]*binEdges[1:])
  if comulative:
    massFunction = (np.sum(hist) - np.cumsum(hist)) / boxSize**3
  else:
    massFunction = hist/ boxSize**3
  return [binCenters, massFunction]
