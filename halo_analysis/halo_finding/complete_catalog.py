import sys, os, time
import subprocess




dataDir = '/raid/bruno/data/cosmo_sims/gadget/ay9_256/halos/'
devDirectory = '/home/bruno/Desktop/Dropbox/Developer/'

rockstarDir = devDirectory + 'cosmo_sims/halo_analysis/halo_finding/rockstar/'
parents_command = rockstarDir + 'util/find_parents'

Lbox = 50 #Mpc/h

header = 'ID DescID Mvir Vmax Vrms Rvir Rs Np X Y Z VX VY VZ JX JY JZ Spin rs_klypin Mvir_all M200b M200c M500c M2500c Xoff Voff spin_bullock b_to_a c_to_a A[x] A[y] A[z] b_to_a(500c) c_to_a(500c) A[x](500c) A[y](500c) A[z](500c) T/|U| M_pe_Behroozi M_pe_Diemer Halfmass_Radius PID'


for nSnap in range(100,101):

  input_file = dataDir + 'out_{0}.list'.format( nSnap )  
  print "\nLoading File: ", input_file

  p = subprocess.Popen([parents_command, input_file, str(Lbox)], stdout=subprocess.PIPE,   stderr=subprocess.PIPE)
  output, err = p.communicate()


  data = output.split('\n')
  halos_data = header
  n_halos = 0
  for line in data:
    if len( line ) == 0: continue
    if line[0] == '#': continue
    halos_data += line + '\n'
    n_halos += 1
    
  print ' N Halos: ', n_halos
  
  output_file = dataDir + 'catalogs/halos_{0:03}.ascii'.format(nSnap)
  print " Writing File: ", output_file

  f = open(output_file, 'w')
  f.write( halos_data )
  f.close()