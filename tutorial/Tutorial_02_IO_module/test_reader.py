### Tutorial 02 IO module


#### Description: 
"""
This tutorial demonstrates how to use functions in IO module,
which interfaces with fortranAPI library to read the dcd, xyz, and txt 
trajectory.
* Part 1: Loading the .txt files (benchmark against np.loadtxt)
* Part 2: Loading the .dcd files (single or multiple configurations) 
* Part 3: Loading the .xyz files (single or multiple configurations) 
* Part 4: Loading the .xyz or .dcd files (use the file extension to choose reader) 
"""

# Python standard library: 
import numpy as np  
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
print ("Numpy loadtxt took: %.7f s"%time_took)

#### Part 2. Loading the .dcd data (Load single or multiple configurations) 

dcdfile = "traj.dcd"
# The total number of atoms and total number of configurations in the trajectory
total_atoms, total_frames = reader.call_read_dcd_header(dcdfile)
# Load a single configuration:
# the 5th configuration in the trajectory
current_frame = 5 
# "xyz" is a 3-dimension (N X 3) cartesian coordinate matrix; N is the total number of atoms
# "box" is the 3-dimensions array of system box size (cubic box assumed)
# "return_numpy=True": the returned arrays are numpy or ctypes array
xyz, box = call_read_dcd_xyz_box(dcdfile, current_frame, total_atoms, return_numpy=True)
print ("xyz coordinate matrix: ", xyz)
print ("box size: ", box)
# Load multiple configurations at once:
# "start_at": which configuration to start in a trajectory
# "num_configs": how many configurations to be read into memory at once starting from "start_at"
# "start_at" will be included in "num_configs"
# "xyz" has a dimension of (M * N * 3), M is the configuration, N is the number of atoms in each configuration
# "box" has a dimension of (M * 3)
xyz, box = call_read_dcd_xyz_box_in_chunk(dcdfile, start_at=1, num_configs=20, total_atoms, return_numpy=True)
print ("xyz coordinate matrix: ", xyz)
print ("box size: ", box)


#### Part 3. Loading the .xyz data (single or multiple configurations)


# Loading xyz file with no box:
# Load a single configuration:
xyzfile = "chunk_no_box.xyz"
total_atoms,total_frames = reader.call_read_xyz_header(xyzfile) 
# read 40th configuration in the trajectory 
current_frame = 40 
# No box information in some types of xyz trajectory files
xyz = reader.call_read_xyz_xyz_box(xyzfile,current_frame,total_atoms,read_box=False,return_numpy=True) 
# Load multiple configurations at once:
start = 1 
num_configs = 401 
xyz = reader.call_read_xyz_xyz_box_chunk(xyzfile,total_atoms,start,num_configs,read_box=False,return_numpy=True) 

### Loading xyz file with box:
# Load a single configuration:
xyzfile = "chunk_with_box.xyz"
total_atoms,total_frames = reader.call_read_xyz_header(xyzfile)
# read 1st configuration in the trajectory
current_frame = 1 
xyz,box= reader.call_read_xyz_xyz_box(xyzfile,current_frame,total_atoms,read_box=True,return_numpy=True) 
# Load multiple configurations at once:
start = 1 
num_configs = 10 
xyz,box = reader.call_read_xyz_xyz_box_chunk(xyzfile,total_atoms,start,num_configs,read_box=True,return_numpy=True)

#### Part 4. Loading either the .xyz or .dcd trajectory:

#xyzfile="/project/palmer/Jingxiang/Monte_Carlo/nvt.xyz"
dcdfile="traj.dcd"
total_atoms,total_frames = reader.call_read_header(dcdfile) 
# Load a single configuration: 
current_frame = 2
xyz,box = reader.call_read_traj(dcdfile,current_frame,total_atoms,return_numpy=True)
# Load multiple configurations at once:
start = 2 
num_configs = 20 
xyz,box = reader.call_read_chunk_traj(dcdfile,start,num_configs,total_atoms,return_numpy=True) 
