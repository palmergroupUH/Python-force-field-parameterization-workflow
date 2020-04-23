# Python standard library 
import numpy as np 
import sys 
import itertools 
import ctypes 
import logging 
from ctypes import CDLL, POINTER, c_int, c_double,c_char_p,c_long,c_float,byref 

# Local library: 
from type_conversion import string_to_ctypes_string, int_to_ctypes_int, np_to_ctypes_array  

# Third-party library: 

dcd_lib = CDLL("./fortran_dcd_reader.so") 

txt_lib = CDLL("./fortran_txt_reader.so")

#-------------------------------------------------------------------------
#                         Parallel workload manager                       
#-------------------------------------------------------------------------

def parallel_assignment(first,last,buffer_size): 

    total_workload = last - first + 1 

    load_times = int(total_workload/buffer_size) + 1  

    work_load_ary =  np.zeros(load_times) 
    
    # every core used no more than buffersize of frames  

    work_load_ary[:] = buffer_size 

    # the left-over (remainder) workload will be sent to the last core:  
    # So, the last core may have workload < buffersize 

    work_load_ary[-1] = total_workload%buffer_size  

    work_load_ary = work_load_ary[work_load_ary != 0] 
 
    pointer_ary = np.zeros(np.count_nonzero(work_load_ary))  
     
    pointer_ary[0] = first 
   
    for i in range(pointer_ary.size-1):

        pointer_ary[i+1] = work_load_ary[i] + pointer_ary[i] 

    return pointer_ary,work_load_ary 

#-------------------------------------------------------------------------
#                          Python Read  LAMMPS traj                       
#-------------------------------------------------------------------------

lammps_traj_header_length = 9 

def read_LAMMPS_traj_as_iterator(fileaddress,start,end,n_col_selected,col_start,col_end):  

    with open(fileaddress,"r") as itear: 

        content = itertools.islice(itear,start,end) 

        for each_line in content:  

            linedata = each_line.split() 

            if ( len(linedata)== n_col_selected ):

                yield linedata[col_start:col_end]

def read_LAMMPS_traj(dtype,fileaddress,start,end,n_col_selected,col_start,col_end): 

    read_LAMMPS = logging.getLogger(__name__) 

    data_itera = read_file_as_iterator(fileaddress,start,end,n_col_selected,col_start,col_end) 

    if ( dtype == "double"):  

        return np.fromiter(itertools.chain.from_iterable(data_itera),dtype=np.float64)

    elif ( dtype == "single"):  

        return np.fromiter(itertools.chain.from_iterable(data_itera),dtype=np.float32)

    else: 

        read_LAMMPS.info("dtype should be either 'single' or 'double' ")        

        sys.exit("Check errors in the log file!") 

    return None 

#-------------------------------------------------------------------------
#                          Fortran dcd reader                             
#-------------------------------------------------------------------------

def call_read_dcd_header(dcdfile): 

    # declare c types varibles: 

    dcdfile,strlength = string_to_ctypes_string(dcdfile)

    total_atoms = c_int() 
    
    total_frames = c_int() 

    dcd_lib.call_dcd_header(dcdfile,byref(strlength),byref(total_atoms),byref(total_frames))  
 
    return total_frames.value, total_atoms.value 

def call_read_xyz_box(dcdfile,current_frame,total_atoms,return_numpy): 

    # declare variables:    

    dcdfile,strlength = string_to_ctypes_string(dcdfile) 

    current_frame = int_to_ctypes_int(current_frame)

    total_atoms = int_to_ctypes_int(total_atoms) 

    #box = (c_double*3)()
    box = np.ctypeslib.as_ctypes(np.zeros(3,dtype=np.float64)) 
    
    #xyz = ((c_float*total_atoms.value)*3)() 

    xyz = np.ctypeslib.as_ctypes(np.zeros(total_atoms.value*3,dtype=np.float32)) 

    #current_frame = c_int(current_frame) 

    dcd_lib.call_dcd_traj(dcdfile,byref(strlength),byref(total_atoms),byref(current_frame),box,xyz) 

    if ( return_numpy ): 

        xyz = np.ctypeslib.as_array(xyz).reshape((total_atoms.value,3)) 

        box = np.ctypeslib.as_array(box)
        
        return xyz,box 
    
    else: 

        return xyz,box 

def call_read_xyz_box_in_chunk(dcdfile,start_at,num_configs,total_atoms,return_numpy): 

    # declare variables:    
    
    dcdfile,strlength = string_to_ctypes_string(dcdfile)  

    num_configs = int_to_ctypes_int(num_configs) 

    start_at = int_to_ctypes_int(start_at) 

    total_atoms = int_to_ctypes_int(total_atoms)

    #box = ((c_double*3)*num_configs)()
    box = np.ctypeslib.as_ctypes(np.zeros(3*num_configs.value,dtype=np.float64)) 

    #xyz = (((c_float*total_atoms.value)*3)*num_configs.value)()  
    xyz = np.ctypeslib.as_ctypes(np.zeros(3*num_configs.value*total_atoms.value,dtype=np.float32))
    
    dcd_lib.call_dcd_traj_chunk(dcdfile,
                                    byref(strlength),
                                    byref(start_at),
                                    byref(num_configs),
                                    byref(total_atoms),
                                    box,
                                    xyz) 

    if ( return_numpy ): 

        xyz = np.ctypeslib.as_array(xyz).reshape((num_configs.value,total_atoms.value,3))

        box = np.ctypeslib.as_array(box).reshape((num_configs.value,3)) 

        return xyz,box 
   
    else: 

        return xyz,box 

#-------------------------------------------------------------------------
#                          Fortran txt reader                             
#-------------------------------------------------------------------------

def get_number_lines(txtfile):  

    txtfile,strlength = string_to_ctypes_string(txtfile)  

    num_lines = c_int() 
    
    txt_lib.get_txt_lines(txtfile,byref(strlength),byref(num_lines)) 

    return num_lines.value 

def loadtxt(txtfile,num_lines,skiprows,return_numpy): 

    txtfile,strlength = string_to_ctypes_string(txtfile)

    num_selected = num_lines-skiprows

    num_lines = int_to_ctypes_int(num_lines)

    skiprows = int_to_ctypes_int(skiprows)

    loaded_data = np.ctypeslib.as_ctypes(np.zeros(num_selected,dtype=np.float64)) 

    txt_lib.load_txt(txtfile,byref(strlength),byref(num_lines),byref(skiprows),loaded_data) 

    if ( return_numpy ): 

        return np.ctypeslib.as_array(loaded_data)

    else: 

        return loaded_data  

