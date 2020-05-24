# Python standard library: 
import numpy as np 
import ctypes 
import os 
import multiprocessing as mp 

# Local library: 
import IO.reader 
import objective.rdf_matching 

# Third-party libraries: 

fortranlib_address =  os.path.join(os.path.dirname(objective.rdf_matching.__file__),"fortran") 

rdf_lib = ctypes.CDLL(os.path.join(fortranlib_address,"fortran_compute_rdf.so")) 

class RadialDistribution():  

    def __init__(self,dcdfile,num_cores,cutoff,num_bins,buffersize,fixed_atoms=True): 
   
        if (fixed_atoms is True): 

           self.total_frames,self.total_atoms = IO.reader.call_read_dcd_header(dcdfile)  
  
        self.dcdfile = dcdfile  

        self.num_bins = num_bins 
    
        self.cutoff = cutoff 
    
        start_at = 1 

        # compute rdf bins:
        self.compute_rdf_bins()

        # create workers for multiprocessing

        self.workers = mp.Pool(num_cores)

        # build parallel workflow 

        self.workflow = IO.reader.parallel_assignment(start_at,self.total_frames,num_cores,buffersize) 

        return None 

    def generate_histogram_job_lst(self):

        job_lst = [] 

        return_numpy = True

        for each_core in self.workflow:
        
            for start_at,num_configs in each_core:
                
                job_lst.append(self.workers.apply_async(RadialDistribution.compute_histogram_in_chunks,
                                                        args=(self.dcdfile,
                                                        start_at,
                                                        num_configs,
                                                        self.total_atoms,
                                                        self.cutoff,
                                                        self.num_bins,
                                                        return_numpy)))

        
        return job_lst 

    @staticmethod
    def compute_histogram_in_chunks(
                                    dcdfile,
                                    start_at,
                                    num_configs,
                                    total_atoms,
                                    cutoff,
                                    num_bins,
                                    return_numpy=False):  
        
        xyz,box = IO.reader.call_read_xyz_box_in_chunk(dcdfile,start_at,num_configs,total_atoms,return_numpy=False)
       
        rdf_histogram = np.ctypeslib.as_ctypes(np.zeros(num_bins,dtype=np.float64))  

        # declare the variable as ctypes:
        total_atoms = ctypes.c_int(total_atoms)

        cutoff = ctypes.c_double(cutoff)         

        num_bins = ctypes.c_int(num_bins) 

        num_configs = ctypes.c_int(num_configs) 

        sum_volume = 0 

        sum_volume = ctypes.c_double(sum_volume) 

        rdf_lib.call_homo_pair_distance_histogram_in_chunk(ctypes.byref(total_atoms),
                                                      ctypes.byref(num_configs),
                                                      ctypes.byref(cutoff),
                                                      ctypes.byref(num_bins),
                                                      xyz,
                                                      box,
                                                      rdf_histogram,
                                                      ctypes.byref(sum_volume)) 
       
        if ( return_numpy is True): 

            return np.ctypeslib.as_array(rdf_histogram), sum_volume.value

        else: 

            return rdf_histogram,sum_volume.value  

    def normalize_histogram(self,job_lst):
    
        sum_hist = np.zeros(self.num_bins)  

        sum_volume = 0

        for job in job_lst:
            
            hist_chunk,volume_chunk = job.get()

            sum_hist += hist_chunk
           
            sum_volume += volume_chunk 

        # convert all into ctyeps: 
        
        bulk_density = ctypes.c_double(self.total_atoms*self.total_frames/sum_volume) 

        num_bins = ctypes.c_int(self.num_bins) 

        cutoff = ctypes.c_double(self.cutoff) 

        total_atoms = ctypes.c_int(self.total_atoms)

        total_frames = ctypes.c_int(self.total_frames )
        
        # declare the ctypes array:
        sum_hist = np.ctypeslib.as_ctypes(sum_hist) 

        gr = np.ctypeslib.as_ctypes(np.zeros(self.num_bins,dtype=np.float64)) 

        rdf_lib.call_normalize_histogram(sum_hist,
                                    ctypes.byref(num_bins),
                                    ctypes.byref(cutoff),
                                    ctypes.byref(total_atoms),
                                    ctypes.byref(total_frames),
                                    ctypes.byref(bulk_density),
                                    gr) 
        
        # Terminate and clean workers: 

        self.workers.close() 

        self.workers.join()

        self.gr = np.ctypeslib.as_array(gr)  
    
        return self.gr  

    def compute(self): 

        job_lst = self.generate_histogram_job_lst()

        return self.normalize_histogram(job_lst)  

    def compute_rdf_bins(self):

        bin_interval = self.cutoff/self.num_bins 

        self.rdf_bins = np.zeros(self.num_bins) 

        for i in range(self.num_bins): 

            self.rdf_bins[i] = bin_interval*i + bin_interval*0.5  

        return None      
        
    def dump_gr(self,filename):
    
        with open(filename,"w") as output:
        
            np.savetxt(output,np.c_[self.rdf_bins,self.gr]) 
            
        return None  

    def dump_r2hr(self,filename): 

        r2hr = self.rdf_bins**2*(self.gr-1)

        with open(filename,"w") as output: 

            np.savetxt(output,np.c_[self.rdf_bins,r2hr]) 

        return None 
