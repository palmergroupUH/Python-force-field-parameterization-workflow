# Python standard library:
import numpy as np 
import multiprocessing as mp  
import sys 
import logging
import time 
import os 
import itertools 
import shutil

# local library:  
import IO.check_file 
import IO.check_type
import IO.reader 
import IO.user_provided 

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

        # parse the objective, cores information: 

        self.parse_argument_dict(argument_dict)
    
        # parse the user-defined input information: 

        self.parse_user_defined(argument_dict) 

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

        self.predicted_address_lst = [] 

        for ref_address,predict_address in zip(ref_address_tple,predit_address_tple): 

            # get Reference energy and force address: 
            
            ref_energy_file = os.path.join(ref_address,self.Ref_energy_file)  

            ref_force_file = os.path.join(ref_address,self.Ref_force_file)

            predict_energy_file = os.path.join(predict_address,self.predict_energy_file)

            predict_force_file = os.path.join(predict_address,self.predict_force_file)

            self.predicted_address_lst.append(predict_address)  

            self.Pre_load_energy_data(ref_energy_file) 

            IO.check_file.status_is_ok(ref_energy_file)

            IO.check_file.status_is_ok(ref_force_file) 

            num_lines_eng,num_colums = IO.reader.get_lines_columns(ref_energy_file) 
    
            num_lines_force,num_colums = IO.reader.get_lines_columns(ref_force_file) 

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

            predicted_num_lines,column = IO.reader.get_lines_columns(predicted_data)
        
            if (predicted_num_lines == self.ref_force_lines[i]): 
            
                self.logger.info("Predicted force data is ready  ... ")

                return None 

            else: 

                self.wait_for_data_to_be_ready(predicted_num_lines,self.ref_force_lines[i]) 
    
        return None 

    def wait_for_data_to_be_ready(self,predicted_data,ref_force_lines):

        count_time = 0 

        self.logger.info("Waiting for predicted force data ... ") 

        while True: 

            predicted_num_lines,num_columns = IO.reader.get_lines_columns(predicted_data) 

            if ( predicted_num_lines !=  ref_force_lines ):
        
                time.sleep(5) 

                self.logger.info("time elapsed: %d ... \n"%count_time)

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

    # parse mandatory user-input: 
    def parse_argument_dict(self,argument):  
        # argument is a tuple
    
        self.sub_folder = argument[0]
        
        # convert objective weight into float 
        
        self.obj_weight = float(argument[1]) 

        # convert cores for analysis into integer

        self.num_cores = int(argument[3] ) 

        # equally assign cores for processing predicted and reference data 
        self.num_cores_ref = int(self.num_cores/2) 

        self.num_cores_predict = int(self.num_cores/2) 

        return None 

    # parse the user-defined input 
    def parse_user_defined(self,argument): 

        # --------------- user defined argument ----------------------
        # user defined: "w 1.0 1.0 bf 5000 eng abs virial"
        # get the weight between energy and force 
        
        argument_str = argument[-1].split()  

        self.parse_weight_arg(argument_str) 
        
        self.parse_buffersize_arg(argument_str) 

        self.parse_eng_arg(argument_str) 

        return None 

    def parse_weight_arg(self,argument_str): 

        keyword_index = IO.user_provided.keyword_exists(argument_str,"w") 

        if ( keyword_index < 0 ):  

            self.logger.warn("WARRNING: missing weight 'w' in the force matching argument\n" 
                             "If none, force and energy is assumed to be equally weighted") 

            self.weight_force_eng = np.array([1.0,1.0],dtype=np.float64)

            return None 

        try: 

            self.weight_force_eng = np.array([
                                             float(argument_str[keyword_index+1]),
                                             float(argument_str[keyword_index+2])])  

        except (ValueError,TypeError): 

            self.logger.error("ERROR: type or value errors in choosing weight between force and energy; The format should be 'w float float' ")  

            sys.exit("Check errors in the log file") 

            self.logger.warn("WARRNING: missing weight 'w' in the force matching argument\n"
                             "If none, force and energy is assumed to be equally weighted")  
             
        return None 

    def parse_buffersize_arg(self,argument_str): 

        keyword_index = IO.user_provided.keyword_exists(argument_str,"bf") 

        if ( keyword_index < 0 ):  

            self.logger.error("ERROR: missing buffersize 'bf' in the force matching argument") 

            sys.exit("Check errors in the log file") 

        try: 

            self.buffersize = int(argument_str[keyword_index+1])  

        except ( ValueError,TypeError):

            self.logger.error("ERROR: buffer index argument error; The format is 'bf integer' ") 

            sys.exit("Check errors in the log file") 

        return None 

    def parse_eng_arg(self,argument_str):

        keyword_index = IO.user_provided.keyword_exists(argument_str,"eng") 
    
        if ( keyword_index < 0 ): 

            self.eng_keyword = "var" 

            self.logger.warn("WARRNING: missing engergy matching 'eng' in the force matching argument\n")

            self.logger.warn("if none, 'eng relative' is used instead\n") 

            return None 

        if ( not IO.check_type.is_string(argument_str[keyword_index+1])):  

            self.logger.error("ERROR: energy keyword type error; The keyword is a string;'eng abs' or 'eng var'' ")

            sys.exit("Check errors in the log file") 

        try: 

            self.eng_keyword = argument_str[keyword_index+1] 

        except ( ValueError,TypeError): 

            self.logger.error("ERROR: energy keyword type error; The keyword is a string;'eng abs' or 'eng relative'' ") 

            sys.exit("Check errors in the log file")

        return None 

    def parse_virial_arg(self): 

        keyword_index = IO.user_provided.keyword_exists(argument_str,"virial")

        if (keyword_index < 0):
    
            self.virial_keword = False 

            return None 
    
     

        return None 

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

        if (self.weight_force_eng[1] == 0.0):  

            self.logger.warn("WARNNING: The weight for force matching is 0; skip the force matching\n") 

            return None 

        self.num_congigs_lst = [] 
    
        self.num_atoms_lst = []

        self.ref_force_norm_lst = [] 

        self.workers = mp.Pool(self.num_cores) 

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
                                         buffer_size=self.buffersize,
                                         workers=self.workers) 

            # computing the force normalization :

            self.ref_force_norm_lst.append(self.pre_compute_force_norm(force_ref_jobs,num_configs,num_atoms,num_column=3))

        self.workers.close() 
    
        self.workers.join() 

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

            if (self.weight_force_eng[1] != 0.0): 

                self.ref_workers = mp.Pool(self.num_cores_ref) 

                self.predict_workers = mp.Pool(self.num_cores_predict) 
                
                # launch the job in parallel jobs 
                # start reading reference force data 
                force_ref_jobs = IO.reader.read_LAMMPS_traj_in_parallel(ref_file,
                                             self.num_cores_ref,
                                             self.num_atoms_lst[i],
                                             self.num_congigs_lst[i],
                                             first=1,
                                             buffer_size=self.buffersize,
                                             workers=self.ref_workers) 

                # start reading predicted force data
                force_predict_jobs = IO.reader.read_LAMMPS_traj_in_parallel(predict_file,
                                             self.num_cores_predict,
                                             self.num_atoms_lst[i], 
                                             self.num_congigs_lst[i], 
                                             first=1,
                                             buffer_size=self.buffersize,
                                             workers=self.predict_workers) 

                sum_sqr_diff = 0  
        
                # update the counter 

                for ref_output,predict_output in zip(force_ref_jobs,force_predict_jobs): 

                    sum_sqr_diff  += np.sum(np.square(( ref_output.get() - predict_output.get() ))) 
                
                self.fm_objective_lst.append(sum_sqr_diff/self.ref_force_norm_lst[i]) 

                i += 1 

                self.ref_workers.close()     

                self.predict_workers.close() 

                self.ref_workers.join() 

                self.predict_workers.join() 
        
            else: 

                self.fm_objective_lst.append(0) 
        
        return None  

#----------------------------------------------------------------------------
#                             Energy Matching :                               
#----------------------------------------------------------------------------

    def Initialize_energy_matching(self): 

        # if weight of energy is 0, no need to do energy matching:

        if (self.weight_force_eng[0] == 0.0): 

            self.logger.warn("WARNNING: The weight for energy matching is 0; skip energy matching\n") 

            return None 

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

            energy_norm = np.var(ref_energy_data)

            return ref_energy_data,energy_norm    

    def compute_energy_matching_objective(self):

        self.energy_objective_lst = [] 

        i = 0 

        for ref_file,predict_file in zip(self.Ref_energy_file_lst,
                                           self.predict_energy_file_lst):

            if (self.weight_force_eng[0] != 0.0): 

                predicted_eng_data = IO.reader.loadtxt(predict_file,
                                  self.ref_eng_lines[i]+1,
                                  skiprows=1,
                                  return_numpy=True)

                if (self.eng_keyword == "var"):
               
                    self.energy_objective_lst.append( self.compute_scaled_var_energy(predicted_eng_data,
                                                                                 self.ref_eng_data_lst[i],
                                                                                 self.ref_eng_norm_lst[i])) 

                elif (self.eng_keyword =="abs"):  

                    self.energy_objective_lst.append(self.compute_scaled_abs_energy(predicted_eng_data,
                                                                                    self.ref_eng_data_lst[i],
                                                                                    self.ref_eng_norm_lst[i]))


                else:  

                    self.logger.info("The energy matching keyword not recognized: Choose 'var' or 'abs'")
                    sys.exit("Check errors in the log file !")

                i += 1 
                
            else: 
            
                self.energy_objective_lst.append(0) 

        return None  

    def compute_scaled_var_energy(self,predicted_eng, ref_energy,eng_norm): 

        diff = predicted_eng - ref_energy 

        ave_diff = np.average( diff) 

        relative_eng = (diff -ave_diff)**2 

        return np.average(relative_eng/eng_norm) 

    def compute_scaled_abs_energy(self,predicted_eng,ref_energy,eng_norm):

        return np.average((predicted_eng - ref_energy)**2/eng_norm)


#----------------------------------------------------------------------------
#                             Virial  Matching                               
#----------------------------------------------------------------------------

        
#----------------------------------------------------------------------------
#                             Compute overall objective:                     
#----------------------------------------------------------------------------

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
    
        #print ( "scaled energy: ", scaled_eng_objective ,"scaled force: ", scaled_force_objective )
        return self.obj_weight*( scaled_eng_objective + scaled_force_objective )  

    # output of predicted force and energy data 

    def rename(self,status,output_folder): 

        counter = 0 

        for eng_file,force_file in zip(self.predict_energy_file_lst,self.predict_force_file_lst): 
    
            if (status =="guess"): 

                initia_predicted_force = self.sub_folder+"_guess"+".force"
    
                initia_predicted_eng = self.sub_folder+"_guess"+".eng"

                dest_eng = os.path.join(output_folder,initia_predicted_eng) 

                dest_force = os.path.join(output_folder,initia_predicted_force)

                shutil.move(eng_file,dest_eng)

                shutil.move(force_file,dest_force)

            elif (status =="old"):  

                current_force_file = os.path.join(self.predicted_address_lst[counter],status+".force") 
            
                current_eng_file = os.path.join(self.predicted_address_lst[counter],status+".eng") 

                shutil.copyfile(force_file,current_force_file) 

                shutil.copyfile(eng_file,current_eng_file)

                counter += 1 

        return None 

    def update(self,keyword,output_folder): 

        counter = 0 

        for eng_file,force_file in zip(self.predict_energy_file_lst,self.predict_force_file_lst): 

            predicted_force = self.sub_folder + "_best" + ".force" 
        
            predicted_eng = self.sub_folder + "_best" + ".eng" 

            dest_force = os.path.join(output_folder,predicted_force) 

            dest_eng = os.path.join(output_folder,predicted_eng) 

            if (keyword =="new"): 
            
                shutil.move(eng_file,dest_eng) 
                       
                shutil.move(force_file,dest_force) 
 
            elif ( keyword =="old"): 

                current_force_file = os.path.join(self.predicted_address_lst[counter],keyword+".force") 

                current_eng_file = os.path.join(self.predicted_address_lst[counter],keyword+".eng") 

                shutil.move(current_force_file,dest_force)

                shutil.move(current_eng_file,dest_eng)
    
        return None 
    
