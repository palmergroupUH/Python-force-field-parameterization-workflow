### Tutorial 02 IO module


#### Description: 

This tutorial demonstrates how to use functions in IO module,
which interfaces with fortranAPI library to read the dcd, xyz, and txt 
trajectory. The module "IO.reader" you imported is basically a wrapper to
the fortranAPI IO module.
* Part 1: Loading the .txt file (benchmark against np.loadtxt)
* Part 2: Loading the .txt traj (e.g. lammps.traj)
* Part 3: Loading the .dcd file (single or multiple configurations) 
* Part 4: Loading the .xyz file (single or multiple configurations) 
* Part 5: Automatic dectection of .dcd or .xyz files (use the file extension to choose reader) 


# Python standard library: 
import numpy as np  
import multiprocessing as mp
import time 

# Local library: 
from IO import reader

#### Part 1. Loading the .txt data (single or multiple columns)

##### Load single column data file (IO.reader.np_loadtxt):

# read 500 configuration of potential energy of in single column file
txtfile = "Ref.eng"
start = time.time()
data = reader.np_loadtxt(txtfile,skiprows=50)
time_took = time.time() - start
print ("IO.reader.np_loadtxt reader took: %.7f s" % time_took)
##### compared with np.loadtxt:  
start = time.time() 
data = np.loadtxt(txtfile,skiprows=50)
time_took = time.time()  - start 
print ("Numpy loadtxt took: %.7f s"%time_took)

##### Load multiple columns data file (IO.reader.np_loadtxt):
txtfile = "Ref.gr" 
start = time.time() 
data = reader.np_loadtxt(txtfile,skiprows=10)
time_took = time.time() - start
print ("IO.reader.np_loadtxt took: %.7f s" % time_took)
##### np.loadtxt: 
start = time.time() 
data = np.loadtxt(txtfile,skiprows=10)
time_took = time.time()  - start 
print ("Numpy loadtxt took: %.7f s" % time_took)

#### Part 2. Loading the .txt trajectory (for reading force data from LAMMPS.traj)

# the trajectory containing intermolecular force data (N x 3) dimensions
# The trajectory has certain formats and some rows or columns have to be skipped
lammpstrj = "Ref.force"
# which configuration to start with
start_at = 1
# number of cores used for multiprocessing
num_cores = 2
# maximum number of configurations read into memory by each core
buffer_size = 1000
# initialize the workers for multiprocessing
workers = mp.Pool(num_cores)
# Step 1: get number of atoms
total_atoms = reader.read_LAMMPS_traj_num_atoms(lammpstrj)
# Step 2: get number of lines:
num_lines, num_columns = reader.get_lines_columns(lammpstrj)
# Step 3: get the number of configurations:
num_configs = reader.get_num_configs_LAMMPS_traj(total_atoms, num_lines)
# Step 4: get the force data from LAMMPS traj
# the "Ref.force" contains 100 configurations
# Jobs are launched through non-blocking "apply_async"
force_data = reader.read_LAMMPS_traj_in_parallel(lammpstrj,
                                                 num_cores,
                                                 total_atoms,
                                                 num_configs,
                                                 start_at,
                                                 buffer_size,
                                                 workers)
# process the data extracted by each core
for each_core in force_data:  
    print ("Force data read by each core:", each_core.get())

#### Part 3. Loading the .dcd data (Load single or multiple configurations) 

# dcd file is a binary file containing box and xyz coordinates in single-precision
dcdfile = "traj.dcd"
# The total number of atoms and total number of configurations in the trajectory
# total_frames = total_configurations here !
total_atoms, total_frames = reader.call_read_dcd_header(dcdfile)
# Load a single configuration:
# the 5th configuration in the trajectory
current_frame = 5 
# "xyz" is a 3-dimension (N X 3) cartesian coordinate matrix; N is the total number of atoms
# "box" is the 3-dimensions array of system box size (cubic box assumed)
# "return_numpy=True": the returned arrays are numpy or ctypes array
xyz, box = reader.call_read_dcd_xyz_box(dcdfile, current_frame, total_atoms, return_numpy=True)
print ("In the 5th configuration, 1st and 100th atoms's coordinates are: ", xyz[0,:], xyz[99,:])
print ("In the 5th configuration, the box size is ", box) 
# Load multiple configurations at once:
# "start_at": which configuration to start in a trajectory
# "num_configs": how many configurations to be read into memory at once starting from "start_at"
# "start_at" will be included in "num_configs"
# "xyz" has a dimension of (M * N * 3), M is the configuration, N is the number of atoms in each configuration
# "box" has a dimension of (M * 3)
start_at=1 
num_configs=20
xyz, box = reader.call_read_dcd_xyz_box_in_chunk(dcdfile, start_at, num_configs, total_atoms, return_numpy=True)
print ("In the 1st configuration, 100th atoms's xyz coordinate is: ", xyz[0,99,:], xyz[0,99,:])
print ("In the 10th configuration, 100th atom's xyz coordinate is: ", xyz[9,99,:], xyz[9,99,:])
print ("In the 1st configuration, the box size is: ", box[0,:])
print ("In the 10th configuration, the box size is: ", box[9,:])

#### Part 4. Loading the .xyz data (single or multiple configurations)


# Loading a xyz file with no box information: a single configuration
xyzfile = "chunk_no_box.xyz"
total_atoms, total_frames = reader.call_read_xyz_header(xyzfile)
# read 40th configuration in the trajectory 
current_frame = 40 
# No box information in some types of xyz trajectory files
xyz = reader.call_read_xyz_xyz_box(xyzfile,current_frame,total_atoms,read_box=False,return_numpy=True) 
print ("In the 40th configuration, 1st and 100th atoms's coordinates are: ", xyz[0,:], xyz[99,:])
print ("No box information from the xyz trajectory")
# Load multiple configurations at once:
start = 1 
num_configs = 401 
xyz = reader.call_read_xyz_xyz_box_chunk(xyzfile,total_atoms,start,num_configs,read_box=False,return_numpy=True) 
print ("In the 40th configuration, 1st and 100th atoms's coordinates are: ", xyz[39,0,:], xyz[39,99,:])
print ("No box information from the xyz trajectory")
### Loading xyz file with box information:
# Load a single configuration:
xyzfile = "chunk_with_box.xyz"
total_atoms,total_frames = reader.call_read_xyz_header(xyzfile)
# read 1st configuration in the trajectory
current_frame = 1 
xyz,box= reader.call_read_xyz_xyz_box(xyzfile,current_frame,total_atoms,read_box=True,return_numpy=True) 
print ("In the 1st configuration, 1st and 100th atoms's coordinates are: ", xyz[0,:], xyz[99,:])
print ("In the 1st configuration, the box size is: ", box)
# Load multiple configurations at once:
start = 1 
num_configs = 10 
xyz,box = reader.call_read_xyz_xyz_box_chunk(xyzfile,total_atoms,start,num_configs,read_box=True,return_numpy=True)
print ("In the 5th configuration, 100th atom's xyz coordinate is: ", xyz[4,99,:], xyz[4,99,:])
print ("In the 5th configuration, the box size is: ", box[4,:])

#### Part 5. Automatic dectection of .xyz or .dcd trajectory:

* Try generic functions: "call_read_header", "call_read_traj", and "call_read_chunk_traj"
* These genric function will automatically switch reader based on file extension
* currently only .xyz and .dcd file is supported. 
* Try dcd file:

dcdfile="traj.dcd"
total_atoms,total_frames = reader.call_read_header(dcdfile) 
# Load a single configuration: 
current_frame = 2
xyz,box = reader.call_read_traj(dcdfile,current_frame,total_atoms,return_numpy=True)
print ("In the 2nd configuration, 1st and 100th atoms's coordinates are: ", xyz[0,:], xyz[99,:])
print ("In the 2nd configuration, the box size is: ", box) 
# Load multiple configurations at once:
start = 2 
num_configs = 20 
xyz,box = reader.call_read_chunk_traj(dcdfile,start,num_configs,total_atoms,return_numpy=True) 
print ("In the 10th configuration, 1st and 100th atoms's coordinates are: ", xyz[9,0,:], xyz[9,99,:])
print ("In the 10th configuration, the box size is: ", box[9,:]) 


# Try xyz file:
# Note that all configurations in "chunk_with_box.xyz" are just a copy of first configuration 
xyzfile="chunk_with_box.xyz"
total_atoms,total_frames = reader.call_read_header(xyzfile) 
# Load a single configuration: 
current_frame = 2
xyz,box = reader.call_read_traj(xyzfile,current_frame,total_atoms,return_numpy=True)
print ("In the 2nd configuration, 1st and 100th atoms's coordinates are: ", xyz[0,:], xyz[99,:])
print ("In the 2nd configuration, the box size is: ", box) 
# Load multiple configurations at once:
start = 2 
num_configs = 20 
xyz,box = reader.call_read_chunk_traj(xyzfile,start,num_configs,total_atoms,return_numpy=True) 
print ("In the 10th configuration, 1st and 100th atoms's coordinates are: ", xyz[9,0,:], xyz[9,99,:])
print ("In the 10th configuration, the box size is: ", box[9,:]) 

