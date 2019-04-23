using HDF5
current_dir = pwd()
cosmo_dir = "/home/bruno/Desktop/Dropbox/Developer/cosmo_sims/"
tools_dir = cosmo_dir * "tools/"
push!(LOAD_PATH, tools_dir)
using CIC_functions

# Input File
dataDir = "/home/bruno/Desktop/data/"
# dataDir = "/home/bruno/Desktop/data/"
inDir = dataDir * "cosmo_sims/gadget/ay9_256/256_dm/h5_files/"
outDir = dataDir * "cosmo_sims/gadget/ay9_256/256_dm/h5_files/"
in_base_name = "snapshot_"
out_base_name = "grid_CIC_"
# out_base_name = "snapshot_"

#Domain Parameters
const x_min = 0.0
const y_min = 0.0
const z_min = 0.0
const x_max = 115e3
const y_max = 115e3
const z_max = 115e3
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

#Gas properties
gamma = 5/3
t = 0
dt = 1e-10
n_step = 0


# #Cells positions in grid ( mid-point )
# c_pos_x = linspace( x_min + dx/2, x_max - dx/2, nx)
# c_pos_y = linspace( y_min + dy/2, y_max - dy/2, ny)
# c_pos_z = linspace( z_min + dz/2, z_max - dz/2, nz)


# nSnap = 0
for nSnap in 100:101
  println( "\nSnapshot: $(nSnap)")
  snapKey = lpad(nSnap,3,0)
  inFileName = inDir * in_base_name * snapKey * ".h5"
  outFileName = outDir * out_base_name * snapKey * ".h5"

  print(" Loading File: $(inFileName)\n")
  inFile = h5open( inFileName, "r")
  # file_name_ramses = "/home/bruno/Desktop/hard_drive_1/ramses_256/h5_files/snapshot_$(snapKey).h5"
  # inFile_ramses = h5open( file_name_ramses, "r")

  current_a = read( attrs(inFile), "current_a" )
  current_z = read( attrs(inFile), "current_z" )
  # current_a_r = read( attrs(inFile_ramses), "current_a" )
  # current_z_r = read( attrs(inFile_ramses), "current_z" )
  # print(" Current_a: $(current_a)\n")
  # print(" Current_a_ramses: $(current_a_r)\n")

  print(" Writing File: $(outFileName)\n")
  outFile = h5open( outFileName, "w")
  attrs(outFile)["current_a"] = current_a
  attrs(outFile)["current_z"] = current_z

  # attrs(outFile)["gamma"] = gamma
  # attrs(outFile)["t"] = t
  # attrs(outFile)["dt"] = dt
  # attrs(outFile)["n_step"] = n_step


  for part_type in [ "dm"]
    # part_type = "dm"
    println(" $(part_type)")
    data = inFile[part_type]
    mass = read( data, "mass" )
    pos_x = read( data, "pos_x" )
    pos_y = read( data, "pos_y" )
    pos_z = read( data, "pos_z" )
    # sort!(pos_x)
    # sort!(pos_y)
    # sort!(pos_z)
    # vel_x = read( data, "vel_x" ) * sqrt( current_a )
    # vel_y = read( data, "vel_y" ) * sqrt( current_a )
    # vel_z = read( data, "vel_z" ) * sqrt( current_a )
    # vel_x = read( data, "vel_x" )
    # vel_y = read( data, "vel_y" )
    # vel_z = read( data, "vel_z" )
    # u = read( data, "u" )
    # data_r = inFile_ramses[part_type]
    # mass_r = read( data_r, "mass" )
    # pos_x_r = read( data_r, "pos_x" )
    # pos_y_r = read( data_r, "pos_y" )
    # pos_z_r = read( data_r, "pos_z" )
    # pos_x_r *= 1.00184
    # pos_y_r *= 1.00184
    # pos_z_r *= 1.00184
    # sort!(pos_x_r)
    # sort!(pos_y_r)
    # sort!(pos_z_r)

    nParticles = size( mass )[1]
    println( "  N parts: $(nParticles)")
    p_inside = ones( Bool, nParticles )
    # get_particles_outside_CIC( p_inside, pos_x, pos_y, pos_z, x_min, x_max, y_min, y_max, z_min, z_max, dx, dy, dz  )
    g_dens = get_interp_CIC( p_inside, mass, mass, mass, pos_x, pos_y, pos_z, nx, ny, nz, x_min, y_min, z_min, x_max, y_max, z_max, dx, dy, dz, true, true )
    # g_velX = get_interp_CIC( p_inside, vel_x, pos_x, pos_y, pos_z, nx, ny, nz, x_min, y_min, z_min, x_max, y_max, z_max, dx, dy, dz, c_pos_x, c_pos_y, c_pos_z, false )
    # g_velY = get_interp_CIC( p_inside, vel_y, pos_x, pos_y, pos_z, nx, ny, nz, x_min, y_min, z_min, x_max, y_max, z_max, dx, dy, dz, c_pos_x, c_pos_y, c_pos_z, false )
    # g_velZ = get_interp_CIC( p_inside, vel_z, pos_x, pos_y, pos_z, nx, ny, nz, x_min, y_min, z_min, x_max, y_max, z_max, dx, dy, dz, c_pos_x, c_pos_y, c_pos_z, false )
    # g_u = get_interp_CIC( p_inside, u, pos_x, pos_y, pos_z, nx, ny, nz, x_min, y_min, z_min, x_max, y_max, z_max, dx, dy, dz, c_pos_x, c_pos_y, c_pos_z, false )
    # vel2 = g_velX.*g_velX + g_velY.*g_velY + g_velZ.*g_velZ
    dens_avrg = mean(g_dens)
    # v2_avrg = mean(vel2)
    # u_avrg = mean(g_u)
    println( "  Dens mean: $(dens_avrg)")


    # nParticles_r = size( mass_r )[1]
    # println( "  N parts_r: $(nParticles_r)")
    # p_inside_r = ones( Bool, nParticles_r )
    # # get_particles_outside_CIC( p_inside_r, pos_x_r, pos_y_r, pos_z_r, x_min, x_max, y_min, y_max, z_min, z_max, dx, dy, dz  )
    # g_dens_r = get_interp_CIC( p_inside_r, mass_r, mass_r, mass_r, pos_x_r, pos_y_r, pos_z_r, nx, ny, nz, x_min, y_min, z_min, x_max, y_max, z_max, dx, dy, dz,  true, true )
    # dens_avrg_r = mean(g_dens_r)
    # println( "  Dens mean_r: $(dens_avrg_r)")



    dens = g_dens
    # mom_x = dens .* g_velX
    # mom_y = dens .* g_velY
    # mom_z = dens .* g_velZ
    # # energy = dens .* ( 0.5*vel2 .* g_u ) #NEED TO ADD INTERNAL ENERGY
    # energy = dens .* ( 0.5*vel2 + 5000 ) #NEED TO ADD INTERNAL ENERGY
    # println( "  vel2 mean: $(v2_avrg)")
    # println( "  u mean: $(u_avrg)")

    outFile["$(part_type)/density"] = dens
    # outFile["momentum_x"] = mom_x
    # outFile["momentum_y"] = mom_y
    # outFile["momentum_z"] = mom_z
    # outFile["Energy"] = energy
  end
  close(inFile)
  close(outFile)
end

# ends
# for nSnap in 0:100
