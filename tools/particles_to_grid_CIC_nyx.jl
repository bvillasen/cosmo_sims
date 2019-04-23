using HDF5
current_dir = pwd()
cosmo_dir = "/home/bruno/Desktop/Dropbox/Developer/cosmo_sims/"
tools_dir = cosmo_dir * "tools/"
push!(LOAD_PATH, tools_dir)
using CIC_functions

# Input File
dataDir = "/raid/bruno/data/"
# dataDir = "/home/bruno/Desktop/data/"
# inDir = dataDir * "cosmo_sims/nyx/256_hydro/run_lya/h5_files/"
inDir = dataDir * "cosmo_sims/nyx/256_dm_50Mpc/h5_files/"
outDir = inDir * "gridFields/"
in_base_name = "snapshot_"
out_base_name = "grid_CIC_"
# out_base_name = "snapshot_"

#Domain Parameters
const x_min = 0.0
const y_min = 0.0
const z_min = 0.0
const x_max = 50000
const y_max = 50000
const z_max = 50000
const Lx = x_max - x_min
const Ly = y_max - y_min
const Lz = z_max - z_min

#Grid Properties
const nPoints = 256
const nx = nPoints
const ny = nPoints
const nz = nPoints
const nCells = nx*ny*nz
const dx = Lx / nx
const dy = Ly / ny
const dz = Lz / nz




# nSnap = 0
for nSnap in 0:30
  println( "\nSnapshot: $(nSnap)")
  snapKey = lpad(nSnap,3,0)
  inFileName = inDir * in_base_name * snapKey * ".h5"
  outFileName = outDir * out_base_name * snapKey * ".h5"

  print(" Loading File: $(inFileName)\n")
  inFile = h5open( inFileName, "r")

  current_a = read( attrs(inFile), "current_a" )
  current_z = read( attrs(inFile), "current_z" )

  print(" Writing File: $(outFileName)\n")
  outFile = h5open( outFileName, "w")
  attrs(outFile)["current_a"] = current_a
  attrs(outFile)["current_z"] = current_z


  for part_type in [ "dm"]
    # part_type = "dm"
    println(" $(part_type)")
    data = inFile[part_type]
    mass = read( data, "mass" )
    pos_x = read( data, "pos_x" )
    pos_y = read( data, "pos_y" )
    pos_z = read( data, "pos_z" )
    mass_total = sum(mass)
    pos_x_max = maximum(pos_x)
    pos_y_max = maximum(pos_y)
    pos_z_max = maximum(pos_z)
    pos_x_min = minimum(pos_x)
    pos_y_min = minimum(pos_y)
    pos_z_min = minimum(pos_z)

    println( " Mass: $mass_total")
    println( " Pos X: $pos_x_min  $pos_x_max ")
    println( " Pos Y: $pos_y_min  $pos_y_max ")
    println( " Pos Z: $pos_z_min  $pos_z_max ")

    nParticles = size( mass )[1]
    println( "  N parts: $(nParticles)")
    p_inside = ones( Bool, nParticles )
    get_particles_outside_CIC( p_inside, pos_x, pos_y, pos_z, x_min, x_max, y_min, y_max, z_min, z_max, dx, dy, dz  )
    g_dens = get_interp_CIC( p_inside, mass, mass, mass, pos_x, pos_y, pos_z, nx, ny, nz, x_min, y_min, z_min, x_max, y_max, z_max, dx, dy, dz, true, true )
    dens_avrg = mean(g_dens)
    println( "  Dens mean: $(dens_avrg)")

    dens = g_dens
    outFile["$(part_type)/density"] = dens
  end
  close(inFile)
  close(outFile)
end
