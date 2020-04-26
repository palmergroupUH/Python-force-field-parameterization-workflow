# Python standard library:
import numpy as np 
import multiprocessing as mp  
import sys 
import logging
import time 
import os 
import itertools 

# local library:  
import IO.check_file 
import IO.reader 

# Third-party library: 

# define global variables: 

# This defines the maximum size to be loaded into memory during initializatioin

class load(): 

    count_jobs = 0    

    max_wait_time = 300 # maximum amount of time waiting for data  (seconds) 
    
    total_file_size_allowed = 1000 # MB 

    def __init__(self,ref_address_tple,predit_address_tple,argument_dict): 
        
        load.count_jobs += 1   
        
        self.logger = logging.getLogger(__name__)

        # load pre-determined file name 

        self.loaded_filename()      

        self.set_file_address_and_check_status(ref_address_tple,predit_address_tple) 

        self.parse_argument_dict(argument_dict)

        self.Initialize_force_matching() 

        self.Initialize_energy_matching()

        return None 
  
    def loaded_filename(self):  
    
        # modify the following file names if needed

        self.Ref_energy_file = "Ref.eng"

        self.Ref_force_file  = "Ref.force"

        self.predict_energy_file = "predict.eng"

        self.predict_force_file = "predict.force"

        return None 

    def set_file_address_and_check_status(self,ref_address_tple,predit_address_tple): 

        self.Ref_force_file_lst = [] 

        self.predict_force_file_lst = []

        self.Ref_energy_file_lst = [] 

        self.predict_energy_file_lst = []

        self.ref_force_lines = [] 

        self.ref_eng_lines = [] 

        for ref_address,predict_address in zip(ref_address_tple,predit_address_tple): 

            # get Reference energy and force address: 
            
            ref_energy_file = os.path.join(ref_address,self.Ref_energy_file)  

            ref_force_file = os.path.join(ref_address,self.Ref_force_file)

            predict_energy_file = os.path.join(predict_address,self.predict_energy_file)

            predict_force_file = os.path.join(predict_address,self.predict_force_file)

            self.Pre_load_energy_data(ref_energy_file) 

            IO.check_file.status_is_ok(ref_energy_file)

            IO.check_file.status_is_ok(ref_force_file) 

            num_lines_eng = IO.reader.get_number_lines(ref_energy_file) 
    
            num_lines_force = IO.reader.get_number_lines(ref_force_file) 

            self.ref_eng_lines.append(num_lines_eng )
            
            self.ref_force_lines.append(num_lines_force) 

            self.Ref_energy_file_lst.append(ref_energy_file)

            self.Ref_force_file_lst.append(ref_force_file)

            self.predict_energy_file_lst.append(predict_energy_file) 

            self.predict_force_file_lst.append(predict_force_file) 

        return None 

    def check_predicted_data_status(self): 

        for i,predicted_data in enumerate(self.predict_force_file_lst):

            IO.check_file.status_is_ok(predicted_data)  

            predicted_num_lines = IO.reader.get_number_lines(predicted_data)
        
            if ( predicted_num_lines == self.ref_force_lines[i] ): 
            
                self.logger.info("Predicted force data is ready  ... ")

                return None 

            else: 

                self.wait_for_data_to_be_ready(predicted_num_lines,self.ref_force_lines[i]) 
    
        return None 

    def wait_for_data_to_be_ready(self,predicted_data,ref_force_lines):

        count_time = 0 

        while True: 

            predicted_num_lines = IO.reader.get_number_lines(predicted_data) 

            if ( predicted_num_lines !=  ref_force_lines ):
        
                time.sleep(5) 

                self.logger.info("Waiting for predicted force data ; time elapsed: %d ... \n"%count_time)

                count_time += 5 

            elif ( count_time > load.max_wait_time ) :

                self.logger.error("Maximum amount of waiting time for predicted force data is reached ( 300s ) ... \n")
                self.logger.error("Current number of lines of predicted force data is"
                                   "%d not equal to that of reference data:%d\n"%(predicted_num_lines,ref_force_lines))
                self.logger.error("Check the file address: %s\n"%predicted_data)
                sys.exit("Check errors in the log file")

        return None 

    def Pre_load_energy_data(self,file_address): 

        # set default preload of Ref force and energy as false: 
        
        self.load_ref_eng = False 

        # check Reference energy file is too big or not 
        
        ref_eng_file_size = IO.check_file.get_file_size(file_address,units="MB")   
        
        if ( ref_eng_file_size < load.total_file_size_allowed ):  
            
            self.load_ref_eng = True  

        return None  

#----------------------------------------------------------------------------
#                             Parse the input:                               
#----------------------------------------------------------------------------

    def parse_argument_dict(self,argument):  
        # argument is a tuple
        

        # convert objective weight into float 
        
        self.obj_weight = float( argument[0] ) 

        # convert cores for analysis into integer

        self.num_cores = int(argument[1] ) 

        # equally assign cores for processing predicted and reference data 
        self.num_cores_ref = int(self.num_cores/2) 

        self.num_cores_predict = int(self.num_cores/2) 

        # --------------- user defined argument ----------------------
        # get the weight between energy and force 
        argument_str = argument[-1].split()  
        
        try: 

            self.weight_force_eng = np.array([float(argument_str[1]),float(argument_str[2]) ] )  

        except (ValueError,TypeError): 

            logger.error("ERROR: type or value errors in get weight between force and energy; The format should be 'w float float' ")  

            sys.exit("Check errors in the log file") 

        # get buffersize ( number of frames ) to be readed into memory   

        try: 

            self.buffersize = int(argument_str[4])  

        except ( ValueError,TypeError):

            logger.error("ERROR: buffer index argument error; The format is 'bf integer' ") 
            
            sys.exit("Check errors in the log file") 

    def print_objective_info(self):
        self.logger.info("Reference data address:  \n")
        self.logger.info("The sub_folder name: %s\n"%sub_folder) 
        self.logger.info("The weight of objective function : %.3f \n"%weight)  
        self.logger.info("Number of cores for running sampling: %d \n"%cores_for_sampling) 
        self.logger.info("Number of cores for computing objective: %d\n"% cores_for_objective) 
        self.logger.info("The other arbitrary argument: %s \n"%argument ) 

        return None 
            
#----------------------------------------------------------------------------
#                             Force Matching :                               
#----------------------------------------------------------------------------

    def Initialize_force_matching(self):

        self.num_congigs_lst = [] 
    
        self.num_atoms_lst = []

        self.ref_force_norm_lst = [] 
   
        for i,force_file_name in enumerate(self.Ref_force_file_lst):

            num_lines = self.ref_force_lines[i] 
    
            num_atoms = IO.reader.read_LAMMPS_traj_num_atoms(force_file_name) 
      
            self.num_atoms_lst.append(num_atoms) 
            
            # get the number of configurations: 
            num_configs = IO.reader.get_num_configs_LAMMPS_traj(num_atoms,num_lines) 
            
            self.num_congigs_lst.append(num_configs) 
            
            force_ref_jobs = IO.reader.read_LAMMPS_traj_in_parallel(force_file_name,
                                         self.num_cores,
                                         num_atoms,
                                         num_configs,
                                         first=1,
                                         buffer_size=self.buffersize) 

            # computing the force normalization :

            self.ref_force_norm_lst.append(self.pre_compute_force_norm(force_ref_jobs,num_configs,num_atoms,num_column=3))

        return None 
       
    def pre_compute_force_norm(self,force_job_list,total_configs,num_atoms,num_column):

        sum_refforce = 0 ;  

        sqr_ave = 0 
    
        # loop over all cores of reading force data 

        for output in force_job_list:  

            # get reference data from current core    

            Reference_data = output.get() 

            sum_refforce = sum_refforce + np.sum(Reference_data)

            sqr_ave = sqr_ave + np.sum(Reference_data*Reference_data)            

        average_sqr = (sum_refforce/(total_configs*num_atoms*num_column))**2 
        
        sqr_average =  sqr_ave/(total_configs*num_atoms*num_column)     

        variances_ref = ( sqr_average - average_sqr )*total_configs*num_atoms*num_column

        return variances_ref 
    
    def compute_force_matching_objective(self): 

        self.fm_objective_lst = [] 

        i = 0 

        for ref_file,predict_file in zip(self.Ref_force_file_lst,
                                         self.predict_force_file_lst):  
            
            # launch the job in parallel jobs 
            # start reading reference force data 
            force_ref_jobs = IO.reader.read_LAMMPS_traj_in_parallel(ref_file,
                                         self.num_cores_ref,
                                         self.num_atoms_lst[i],
                                         self.num_congigs_lst[i],
                                         first=1,
                                         buffer_size=self.buffersize) 

            # start reading predicted force data
            force_predict_jobs = IO.reader.read_LAMMPS_traj_in_parallel(predict_file,
                                         self.num_cores_predict,
                                         self.num_atoms_lst[i], 
                                         self.num_congigs_lst[i], 
                                         first=1,
                                         buffer_size=self.buffersize) 

            sum_sqr_diff = 0  
    
            # update the counter 
            

            for ref_output,predict_output in zip(force_ref_jobs,force_predict_jobs): 

                sum_sqr_diff  += np.sum(np.square(( ref_output.get() - predict_output.get() ))) 

            self.fm_objective_lst.append(sum_sqr_diff/self.ref_force_norm_lst[i]) 

            i += 1 
        return None  

#----------------------------------------------------------------------------
#                             Energy Matching :                               
#----------------------------------------------------------------------------

    def Initialize_energy_matching(self): 

        self.ref_eng_data_lst = [] 

        self.ref_eng_norm_lst = [] 

        for i,ref_eng_file in enumerate(self.Ref_energy_file_lst): 
        
            num_lines = self.ref_eng_lines[i] 
        
            ref_energy_data,energy_norm = self.pre_compute_energy_matching_norm(ref_eng_file,num_lines)

            self.ref_eng_data_lst.append(ref_energy_data)  

            self.ref_eng_norm_lst.append(energy_norm )
            
        return None 

    def pre_compute_energy_matching_norm(self,Ref_eng_file,num_lines_eng): 

        if ( self.load_ref_eng == True ):  

            ref_energy_data = IO.reader.loadtxt(Ref_eng_file,
                                                num_lines_eng,
                                                skiprows=0,
                                                return_numpy=True) 

            energy_norm = np.var(ref_energy_data)*num_lines_eng 

        return ref_energy_data,energy_norm    

    def compute_energy_matching_objective(self):

        self.energy_objective_lst = [] 

        i = 0 

        for ref_file,predict_file in zip(self.Ref_energy_file_lst,
                                           self.predict_energy_file_lst):

            predicted_eng_data = IO.reader.loadtxt(predict_file,
                              self.ref_eng_lines[i]+1,
                              skiprows=1,
                              return_numpy=True)

           
            self.energy_objective_lst.append(np.sum(( predicted_eng_data - self.ref_eng_data_lst[i] )**2)/self.ref_eng_norm_lst[i]) 

            i += 1 
             
        return None  

#----------------------------------------------------------------------------
#                             Load the data:                                 
#----------------------------------------------------------------------------

    def compute_scaled_force(num_cores, 
                             total_configs,
                             chunksize,
                             num_atoms,
                             num_column,
                             dumpfile,
                             Reffile,
                             skip_ref,
                             skip_dump):

        p = mp.Pool(num_cores) 

        datafile = [ Reffile,dumpfile ] 

        num_itera = total_configs/chunksize

        remainder = total_configs%chunksize

        if ( remainder != 0 ): 

                #print "Chunksize has to be divisible by total configurations"
                #print "Chunksize is: ", chunksize, " and total configurations are: ", total_configs
                sys.exit()

        sum_refforce = 0.0 ; sqr_ave = 0.0 ; sum_diff  = 0.0

        datasize = chunksize*( num_atoms + 9 )

        for i in range(num_itera):

            start = i*datasize
         
            end = start + datasize

            # push all jobs into a list 

            results = [ p.apply_async( ReadFileByChunk, args=(data,start,end )) for data in datafile ]

            Ref_chunkdata,Dump_chunkdata = [ array.get() for array in results ]

            sum_diff = sum_diff + ComputeSumSquared(Ref_chunkdata,Dump_chunkdata)
        
            sum_refforce = sum_refforce + np.sum(Ref_chunkdata)

            sqr_ave = sqr_ave + np.sum(Ref_chunkdata*Ref_chunkdata)

        average_sqr = (sum_refforce/(total_configs*num_atoms*num_column))**2

        sqr_average =  sqr_ave/(total_configs*num_atoms*num_column)

        variances_ref = sqr_average - average_sqr

        return sum_diff/variances_ref/(total_configs*num_atoms*num_column) 

    def optimize(self): 

        # before evaluating objective functions 
        self.check_predicted_data_status()

        eng_weight = self.weight_force_eng[0] 

        force_weight = self.weight_force_eng[1]

        scaled_eng_objective = 0 

        scaled_force_objective = 0

        self.compute_force_matching_objective() 

        self.compute_energy_matching_objective()
        
        for e_obj,f_obj in zip(self.energy_objective_lst,self.fm_objective_lst): 

            scaled_eng_objective +=  eng_weight*e_obj 

            scaled_force_objective  += force_weight*f_obj

        print ( scaled_eng_objective, scaled_force_objective )
        
        return self.obj_weight*( scaled_eng_objective + scaled_force_objective )  
        
