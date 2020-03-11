# Standard library 
import numpy as np 
import logging
import multiprocessing as mp 
import os 

# Custom Fortran library 

import reader 
import computerdf 


class RadialDistributionFunction: 

    def __init__(self,dcdfile,cores,cutoff,num_bins,buffersize): 
    
        self.dcdfile = dcdfile 

        self.cutoff = float(cutoff) 

        self.num_bins = num_bins

        self.buffersize = buffersize 
    
        self.natoms,self.total_frames = reader.readdcdheader(dcdfile) 

        self.work_sequence,self.loaded_work = initialize_parallel_config_analysis(self.natoms,self.total_frames,cores,self.buffersize)  
        
        self.process = mp.Pool(cores) 
    
        return None 

    def compute(self): 

        results = [ self.process.apply_async( compute_histogram , 
                                args=(self.work_sequence[job],self.dcdfile,
                                self.cutoff,self.num_bins,self.natoms,self.loaded_work[job]))\
                                for job in xrange(self.work_sequence.size)] 
                
        rdf_histogram_all,sum_atoms,total_volume,frame_count = gather_rdf_output(results,self.num_bins) 
    
        density = computerdf.bulk_density(sum_atoms,total_volume)  

        self.gr = computerdf.normalize_histogram(rdf_histogram_all,self.num_bins,self.cutoff,self.natoms,frame_count,density) 

        return self.gr  
    
    def bins_pos(self):  
        
        r_interval = self.cutoff/self.num_bins 

        bins_position = np.zeros(self.num_bins) 

        for i in xrange(self.num_bins): 

            bins_position[i] = r_interval*0.5 + i*r_interval 

        return bins_position 

    def dump(self,filename):

        bins_position = self.bins_pos() 

        np.savetxt(filename,zip(bins_position,self.gr))  

        return None 

#---------------------------------------------------------------------------------- 
#---------------------------- Module Functions: ----------------------------------- 
#---------------------------------------------------------------------------------- 


#------------------------ Parallel Configurations Processing -----------------------

def initialize_parallel_config_analysis(natoms,num_configs,cores,buffersize): 

    # work_load: an array with number of configurations to be processed by each core 
    work_load = reader.assign_workload_to_workers(cores,num_configs)            

    # set the pointer where each core start reading configuratinos: 
    work_pointer = reader.assign_task_to_workers(cores,work_load) 

    # when buffer size is applied, all configurations ( workload)  can not be loaded 
    # directly unless they are less than buffer size. If workload > buffersize 
    # workload will be broken down as loaded_workflow  
    
    # given a buffersize, how many times needed to load all work 

    loaded_times = work_load/buffersize + 1

    #loaded_work = [ reader.load_work_based_on_buffersize(buffersize,workload,loaded_times) for workload in work_load ] 
    
    loaded_work = determined_loadedtimes(work_load,buffersize)  

    return work_pointer, loaded_work   

def determined_loadedtimes(work_load,buffersize): 

    loaded_work = []  
    
    for i in xrange(work_load.size): 

        work_load_per_core = work_load[i]  
        
        if ( work_load_per_core > buffersize ): 

            load_times = work_load_per_core/buffersize + 1  
            
            work_load_more = np.zeros(load_times,dtype=np.int)
        
            work_load_more[:] = buffersize  

            workload_left = work_load_per_core % buffersize
        
            work_load_more[load_times-1] = workload_left 

            loaded_work.append(work_load_more)  

        else:
            
            loaded_work.append(np.array( [work_load_per_core] ) )   

    return loaded_work      

#------------------------ I/O -----------------------

def file_exists(filename): 

    if ( os.path.exists(filename)): 

        return  True 

    else: 
            
        return  False  
    
def loadtxt(filename,skiprows=None):     
        
    if ( skiprows ):  
        
        pass

    else: 

        skiprows = 0 

    num_lines = reader.getlines(filename) 

    num_configs = num_lines - skiprows

    data = reader.loadtxt(filename,num_lines,skiprows)
    
    return data,num_configs

def initialize_Parsing_LAMMPS_Force_Parallel(cores,natoms,total_frames,buffersize): 

    work_sequence,loaded_work = initialize_parallel_config_analysis(natoms,total_frames,cores,buffersize)
    
    create_workers = mp.Pool(cores) 

    set_up_parallel_work =  [ [ [ start+loaded_work[i][j-1]*j, workload]\
                    for j,workload in enumerate(loaded_work[i]) ]\
                        for i,start in enumerate(work_sequence) ] 

    return set_up_parallel_work,create_workers  

def read_lammpsforce_by_chunk(start_configs,filename,natoms,loaded_work):  

    lines_jumped = reader.jump_lammpsoutput_lines(natoms,start_configs) 

    lines_jumped_bytes = reader.convert_lines_to_bytes(filename,lines_jumped) 
    
    force_chunk = reader.read_lammps_force_output(filename,natoms,lines_jumped_bytes,loaded_work) 
    
    return force_chunk  
    
def read_LAMMPS_force(filename,parallel_work_flow,call_times,workers): 

    launch_list = [] 
    
    for process_time in xrange(call_times): 

        launch_parallel_work = [ workers.apply_async(read_lammpsforce_by_chunk,
                                args=( cores[process_time][0], filename,
                                natoms,cores[process_time][1]) )\
                                for cores in parallel_work_flow ]  

        launch_list.append(launch_parallel_work) 

    return launch_list 

#------------------------------ Radial Distribution -----------------------------------

def compute_gr(dcdfile,cores,cutoff,num_bins,buffersize): 

    natoms,total_frames = reader.readdcdheader(dcdfile)         

    work_sequence,loaded_work = initialize_parallel_config_analysis(natoms,total_frames,cores,buffersize)   

    process = mp.Pool(cores) 

    results = [ process.apply_async( compute_histogram , 
                                args=(work_sequence[job],dcdfile,
                                cutoff,num_bins,natoms,loaded_work[job])) for job in xrange(cores)] 
                
    rdf_histogram_all,sum_atoms,total_volume,frame_count = gather_rdf_output(results,num_bins) 
    
    density = bulk_density(sum_atoms,total_volume)  

    gr = normalize_histogram(rdf_histogram_all,num_bins,cutoff,natoms,frame_count,density) 

    return gr  
    
def compute_histogram(start,dcdfile,cutoff,num_bins,natoms,load_work): 

    frame_count = 0 ; sum_atoms = 0.0 ; total_volume = 0.0 

    rdf_histogram_all = np.zeros(num_bins) 

    for loadedwork in load_work:    
        
        xyz,box = reader.read_dcd_in_chunk(start,loadedwork,dcdfile,natoms) 

        for eachframe in xrange(loadedwork):  
            
            rdf_histogram = computerdf.build_homo_pair_distance_histogram(natoms,cutoff,num_bins,xyz[:,:,eachframe],box[0:3,eachframe])         
            
            total_volume = total_volume + np.prod(box[0:3,eachframe])   

            sum_atoms  = sum_atoms + natoms 

            frame_count = frame_count +  1 
            
            rdf_histogram_all = rdf_histogram_all + rdf_histogram 

        start = start + loadedwork

    return rdf_histogram_all,frame_count,sum_atoms,total_volume 
    
def gather_rdf_output(results,num_bins): 

    rdf_histogram_all = np.zeros(num_bins) 
    
    frame_all = 0  

    atoms_all = 0 

    volume_all = 0 

    for output in results:  
    
        rdf_hist_each_core, frame_each_core,atoms_each_core, volume_each_core  = output.get() 
        
        rdf_histogram_all = rdf_histogram_all + rdf_hist_each_core  
    
        frame_all = frame_all + frame_each_core 
    
        atoms_all = atoms_all + atoms_each_core 

        volume_all = volume_all + volume_each_core          

    return rdf_histogram_all,atoms_all,volume_all,frame_all 

#------------------------------ Density ---------------------------------


#------------------------------ Diffusion ---------------------------------


