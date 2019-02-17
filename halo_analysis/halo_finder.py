import os, sys
import numpy as np


def isLocalMaximum( i, j, k, nx, ny, nz, dens, dens_limit ):
  dens_c = dens[k, j, i]
  if dens_c < dens_limit: return False
  n = 1
  indx_l = ( i - n ) % nx
  dens_l = dens[k, j, indx_l]
  if dens_l > dens_c: return False

  indx_r = ( i + n ) % nx
  dens_r = dens[k, j, indx_r]
  if dens_r > dens_c: return False

  indx_d = ( j - n ) % ny
  dens_d = dens[k, indx_d, i]
  if dens_d > dens_c: return False

  indx_u = ( j + n ) % ny
  dens_u = dens[k, indx_u, i]
  if dens_u > dens_c: return False

  indx_b = ( k - n ) % nz
  dens_b = dens[indx_b, j, i]
  if dens_b > dens_c: return False

  indx_t = ( k + n ) % nz
  dens_t = dens[indx_t, j, i]
  if dens_t > dens_c: return False

  return True


def getLocalMax( dens, dens_limit ):
  nz, ny, nx = dens.shape
  localMax_array = np.zeros_like( dens )
  for k in range(nz):
    for j in range(ny):
      for i in range(nx):
        id = i + j*nx + k*nx*ny
        # if id%2 == 1: continue
        localMax = isLocalMaximum( i, j, k, nx, ny, nz, dens, dens_limit )
        localMax_array[k, j, i] = localMax
  return localMax_array

def getMaxIndexs( dens, dens_limit ):
  localMax_array = getLocalMax( dens, dens_limit )
  maxIndxs = np.where( localMax_array == 1 )
  maxIndxs = np.array( maxIndxs ).T
  return maxIndxs

# def getMaxPositions():
#   maxIndxs = getMaxIndexs()
#   delta = np.array([ dz, dy, dx ])
#   positions =  maxIndxs * delta + 0.5*delta
#   pos_z, pos_y, pos_x = positions.T
#   pos_y = np.abs( pos_y - box_size )
#   return pos_z, pos_y, pos_x



def get_CIC_fractions( x_c, y_c, z_c, x_l, y_l, z_l  ):
  dist_x = x_c-x_l
  dist_y = y_c-y_l
  dist_z = z_c-z_l
  # print dist_x, dist_y, dist_z
  dist = np.sqrt( (dist_x)**2 + (dist_y)**2 + (dist_z)**2 )
  indx_0 = int( np.floor(dist) )
  indx_1 = indx_0 + 1
  fracc_0 = 1 - (dist - indx_0)
  fracc_1 =  1 - fracc_0
  return dist, indx_0, indx_1, fracc_0, fracc_1


def add_slides_z( n, x, y, z, mass_array, nx, ny, nz, dens, dv, halo_ids, halo_id ):
  for k in [ z-n, z+n ]:
    indx_z = k % nz
    for j in range( y-n, y+n+1):
      indx_y = j % ny
      for i in range( x-n, x+n+1 ):
        indx_x = i % nx
        dist, indx_0, indx_1, fracc_0, fracc_1 = get_CIC_fractions( x, y, z, i, j, k)
        mass_local = dens[indx_z, indx_y, indx_x] * dv
        halo_ids[indx_z, indx_y, indx_x] = halo_id
        mass_array[indx_0] += mass_local * fracc_0
        mass_array[indx_1] += mass_local * fracc_1



def add_slides_y( n, x, y, z, mass_array, nx, ny, nz, dens, dv, halo_ids, halo_id ):
  for j in [ y-n, y+n]:
    indx_y = j % ny
    for k in range( z-(n-1), z+(n-1)+1 ):
      indx_z = k % nz
      for i in range( x-n, x+n+1 ):
        indx_x = i % nx
        dist, indx_0, indx_1, fracc_0, fracc_1 = get_CIC_fractions( x, y, z, i, j, k)
        mass_local = dens[indx_z, indx_y, indx_x] * dv
        halo_ids[indx_z, indx_y, indx_x] = halo_id
        mass_array[indx_0] += mass_local * fracc_0
        mass_array[indx_1] += mass_local * fracc_1


def add_slides_x( n, x, y, z, mass_array, nx, ny, nz, dens, dv, halo_ids, halo_id ):
  for i in [ x-n, x+n]:
    indx_x = i % nx
    for j in range( y-(n-1), y+(n-1)+1 ):
      indx_y = j % ny
      for k in range( z-(n-1), z+(n-1)+1 ):
        indx_z = k % nz
        dist, indx_0, indx_1, fracc_0, fracc_1 = get_CIC_fractions( x, y, z, i, j, k)
        mass_local = dens[indx_z, indx_y, indx_x] * dv
        halo_ids[indx_z, indx_y, indx_x] = halo_id
        mass_array[indx_0] += mass_local * fracc_0
        mass_array[indx_1] += mass_local * fracc_1




def find_halos( Lbox, dens, dens_limit, halo_limit ):
  nz, ny, nx = dens.shape
  dx, dy, dz = Lbox/nx, Lbox/ny, Lbox/nz
  dv = dx * dy * dz
  maxIndxs = getMaxIndexs( dens, dens_limit )
  halo_ids = -1 * np.ones_like( dens )
  halo_catalog = []
  halo_id = 0
  for indx in maxIndxs:
    z, y, x = indx
    delta = np.array([ dz, dy, dx ])
    z_center, y_center, x_center = indx * delta + 0.5*delta
    n_max = 20
    mass_array = np.zeros( n_max )
    dens_mean = dens[ z, y, x ] * dv / ( 4. /3 * np.pi * (0.5*dx)**3 )
    if dens_mean < halo_limit: continue
    mass_array[0] = dens[ z, y, x ] * dv
    # print ""
    # print x, y, z
    # print dens_mean, halo_limit
    n = 1
    while dens_mean > halo_limit:
      # print n
      # print mass_array
      add_slides_z( n, x, y, z, mass_array, nx, ny, nz, dens, dv, halo_ids, halo_id )
      add_slides_y( n, x, y, z, mass_array, nx, ny, nz, dens, dv, halo_ids, halo_id )
      add_slides_x( n, x, y, z, mass_array, nx, ny, nz, dens, dv, halo_ids, halo_id )
      volume = 4./3 * np.pi * ( n*dx )**3
      dens_mean = mass_array[:n+1].sum() /volume
      n += 1
      # print dens_mean, halo_limit
    halo_mass = mass_array[:n+1].sum()
    halo_radius = n * dx
    r_vir = (halo_mass / ( 4./3 *np.pi * halo_limit) )**(1./3)
    halo_data = [ n, halo_id, halo_mass, halo_radius, x_center, y_center, z_center, r_vir  ]
    halo_catalog.append( halo_data )
    halo_id += 1
  halo_catalog = np.array( halo_catalog )
  return halo_catalog
