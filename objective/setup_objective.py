import numpy as np 
import os 
import sys 
import logging 
import objective.const_mod
import sampling.MD 
import IO.input_file
import IO.check_type
import IO.check_dir 

class setup(): 

    """ 
    Parse the input files:  
    Check the folder existence: 
    Makeing the working directory:  
    Save all input variables into a single dicionary 

    """ 
    
    # The default folder: 

    default_ref_folder = "../ReferenceData"

    default_prep_folder = "../prepsystem"

    def __init__(self,input_file,
                     total_cores,
                     jobid,
                     log_name=None, 
                     overwrite=None,
                     Ref_folder=None, 
                     prep_folder=None,  
                     skip_lines=None,
                     stop_after=None,
                     packagename=None): 

        # set up the logger: 

        self.set_logger(log_name) 

        # The working folder where the optimization jobs are launched 
        self.HOME = os.getcwd() 

        # define the package to be used:     

        self.initialize_packages(packagename) 

        # determine  where the Reference data folder is and 
        # where the template folder ( with input scripts for running sampling methods ) 
        
        self.set_reference_prep_folder_address(Ref_folder,prep_folder) 
       
        self.set_working_directory_address(jobid) 

        self.setup_working_folders(overwrite) 

        # read the whole input data file 

        self.input_data_dict = IO.input_file.parse(input_file,skip_lines,stop_after)  

        # read the 

        self.lastline = self.go_through_each_line()
         
        self.generate_objective_dict(total_cores)

        return None 

    def initialize_packages(self,packagename):

        # define the package to be used:     
        # Default is LAMMPS so far: 

        if ( packagename is None ):  
        
            self.packagename = "LAMMPS"
    
        else:

            self.packagename = packagename
    
        return None 

    def print_initialization_properties_matching(self,
                                                 matching_type,
                                                 sub_folder,
                                                 weight, 
                                                 cores_for_sampling,    
                                                 cores_for_objective,
                                                 argument):

        self.objective_logger.info(29*"-"+" Initialize %s matching "%matching_type+ 29*"-"+"\n\n") 
        self.objective_logger.info("The sub_folder name: %s\n"%sub_folder) 
        self.objective_logger.info("The weight of objective function : %.3f \n"%weight)  
        self.objective_logger.info("Number of cores for running sampling: %d \n"%cores_for_sampling) 
        self.objective_logger.info("Number of cores for computing objective: %d\n"% cores_for_objective) 
        self.objective_logger.info("The other arbitrary argument: %s \n"%argument ) 

        return None 

    def go_through_each_line(self):  

        last_line = [] 
    
        self.matching_type = []   
       
        self.command_lst = [] 

        counter_matching  = 0  

        counter_command  = 0    

        for readline,line_number in enumerate(self.input_data_dict.keys()):   
    
            each_line = self.input_data_dict[line_number]
            
            if ( self.This_line_is_units(each_line,readline) ): 

                self.units_arg = each_line 
                
                last_line.append(line_number) 

                continue 
            
            if ( self.This_line_is_matching_type(each_line) ): 
                
                self.matching_type.append( each_line )   

                last_line.append(line_number) 
    
                counter_matching  +=1   

                continue 

            if ( self.This_is_command_argument(each_line)): 

                self.command_lst.append(" ".join(each_line)) 

                last_line.append(line_number)

                counter_command  += 1   

                continue  

            if (self.This_is_line_is_output_freq(each_line)):  

                break 

        if ( counter_matching == 0 ): 

            self.objective_logger.error("ERROR: missing matching type argument or check its format")             

            sys.exit("Check errors in the log file")

        if ( counter_command == 0 ) :
    
            self.objective_logger.error("ERROR: missing command to run sampling ( MD or MC ) in the input file ")             

            sys.exit("Check errors in the log file")

        self.check_command() 

        return max(last_line)  
 
    def set_logger(self,logname): 
        
        if ( logname is not None ):  

            self.objective_logger = logging.getLogger(__name__)
        
            self.objective_logger.setLevel(logging.INFO)              
    
            fh = logging.FileHandler(logname,mode="w") 

            formatter = logging.Formatter("%(message)s") 
            
            fh.setFormatter(formatter) 

            self.objective_logger.addHandler(fh)

        else:
            
            self.objective_logger = logging.getLogger(__name__) 

        return None 

    # This part must be changed if output,restart frequency format are changed !  

    def This_is_line_is_output_freq(self,each_line):

        if ( len(each_line) == 2 

            and IO.check_type.is_int(each_line[0])

            and IO.check_type.is_int(each_line[1]) ):

            return True  

        else:

            return False

    def This_line_is_units(self,each_line,first_line): 

        if ( len(each_line) == 1 

            and IO.check_type.is_string(each_line[0]) ):  

            self.units = objective.const_mod.Units(each_line[0]) 

            return True 

        elif ( first_line != 0 ): 

            return False
        
        else: 

            self.objective_logger.error("The first line of input file is" 
                                       "not units; Choose the following" 
                                       "units: metal, real ")

            sys.exit("Check errors in log file ! ") 
            
    def This_line_is_matching_type(self,each_line):  
        
        if ( len(each_line) < 5  ):  

            return False

        line_type = np.array([IO.check_type.is_string(each_line[0]), 
                    IO.check_type.is_string(each_line[1]), 
                    IO.check_type.is_float(each_line[2]) , 
                    IO.check_type.is_int(each_line[3]), 
                    IO.check_type.is_int(each_line[4])])  

        # if each line type meet the requirement: 
        
        if ( np.all(line_type)):   

            return True 

        elif ( line_type[1] == False and np.sum(line_type)== 4 ):
            
            self.objective_logger.warning("WARRANING:The second argument for objective instances" 
                                       " must be string ( subfolder_name )") 
        
            return True  

        else: 

            return False

    def This_is_command_argument(self,each_line): 

        # first two argument can not be integer or float:  

        line_is_string = np.array([IO.check_type.is_string(each_line[0]),  
                                  IO.check_type.is_string(each_line[1])])

        if ( np.all(line_is_string)):  

            return True 
            
        else:

            return False
       
    def check_command(self): 

        if ( len(self.command_lst) > 1  ):   

            self.objective_logger.error("ERROR: currently, only 1 command options"
                                        " is supported ( %d commands given ) ;"
                                        " this command will be used"
                                        " in running sampling for all objective" 
                                        " instances"%len(self.command_lst))

            sys.exit("Check errors in log file !") 
            
        return None 

    def check_all_cores(self,
                        total_cores,
                        total_cores_by_sampling,
                        total_cores_by_objective):

        if ( total_cores < total_cores_by_sampling ):   

            self.objective_logger.error("ERROR: total number of cores ( %d )"
                                        " used by sampling method is more than"
                                        " the cores ( %d ) assigned by slurm"%(
                                        total_cores_by_sampling,total_cores))         
         

            sys.exit("Check errors in log file !") 

        if ( total_cores < total_cores_by_objective): 

            self.objective_logger.error("ERROR: total number of cores ( %d )"
                                        "used by objective function evaluations"
                                        " is more than the cores ( %d )assigned by slurm"%(
                                        total_cores_by_objective,total_cores))
        
            sys.exit("Check errors in log file !")    

        return None 

#----------------------------------------------------------------------------
#                         set up the working folder:                         
#----------------------------------------------------------------------------

    def set_reference_prep_folder_address(self,Ref_folder,prep_folder): 
       
        if ( Ref_folder is None ):
        
            self.reference_folder_address = setup.default_ref_folder  
            
        else:

            self.reference_folder_address = Ref_folder
            
            IO.check_dir.decide_folder_status(Ref_folder)  
                
        if ( prep_folder is None ): 

            self.prep_folder_address = setup.default_prep_folder 

        else: 

            self.prep_folder_address = prep_folder 
    
            IO.check_dir.decide_folder_status(prep_folder) 

        return None 

    def set_working_directory_address(self,jobid):     

        # built-in names and address: modify the name for your need  

        predict_folder_name = "Predicted"

        output_folder_name = "Output"

        restart_folder_name = "Restart"
    
        # predicted folder address: 

        self.predicted_folder = os.path.join(jobid, predict_folder_name) 

        # output folder address:  

        self.output_folder = os.path.join(jobid, output_folder_name) 

        # restart folder address: 

        self.restart_folder = os.path.join(jobid, restart_folder_name) 

        return None 

    def dectect_duplicated_folder(self,type_subfolder): 
     
        if ( len(self.matching_type) > len(set(type_subfolder)) ):

            self.objective_logger.error("ERROR: found repeated objective type and its subfolder in input file: !" )
        
            sys.exit("Check errors in log file !")  

        return None 

    def setup_working_folders(self,overwrite): 
   
        # make these folders: 

        # default overwrite is False: avoid overwriting current working folder 

        if ( overwrite is None ):  

            overwrite = False 

        IO.check_dir.mkdirs_if_not_exist(self.predicted_folder,overwrite) 

        IO.check_dir.mkdirs_if_not_exist(self.output_folder,overwrite) 

        IO.check_dir.mkdirs_if_not_exist(self.restart_folder,overwrite) 
       
        return None 

    def transfer_prepsystem_to_working_folders(self,matching_type,subfolder): 
       
        prep_path = os.path.join(self.prep_folder_address,
                                 matching_type,
                                 subfolder) 

        IO.check_dir.decide_folder_status(prep_path,"Job template") 

        predict_folder_type = os.path.join(self.predicted_folder,
                                           matching_type,
                                           subfolder) 

        
        os.system("mkdir -p %s"%predict_folder_type)

        prep_system_content = os.path.join(prep_path,"*") 
        
        os.system("cp -r %s %s"%(prep_system_content,predict_folder_type) ) 

        return None 

    # traverse a folders and its all subfolders 

    def Go_to_Subfolders(self,sub_folders): 

        folders = next(os.walk('.'))[1] 

        if ( folders ): 

            for folder in folders: 

                os.chdir(folder) 

                self.Go_to_Subfolders(sub_folders) 

                os.chdir("../")

        else: 

            sub_folders.append(os.getcwd()) 

        return sub_folders

    # given a folder: traverse its all sub folders: 

    def Get_Path(self,wk_folder): 

        os.chdir(wk_folder) 

        sub_folders = []  

        sub_folders = self.Go_to_Subfolders(sub_folders) 

        num_folders = len(sub_folders)      

        os.chdir(self.HOME) 

        return num_folders,sub_folders 

    def get_objective_input(self,matching_arg): 

        matching_type = matching_arg[0] 

        sub_folder = matching_arg[1]

        weight= float(matching_arg[2] )

        cores_for_sampling = int(matching_arg[3]) 

        cores_for_obj = int(matching_arg[4]) 

        return matching_type,sub_folder,weight, cores_for_sampling, cores_for_obj 

    def generate_objective_dict(self,total_cores): 
    
        self.ref_address_dict = {}   

        self.predict_address_dict = {}   
    
        self.argument_dict = {} 

        type_subfolder = [] 

        total_cores_by_sampling = 0 

        total_cores_by_obj = 0

        for matching_arg in self.matching_type:  
       
            # ----------- print and convert all input parameters ----------------  
            matching_argument = " ".join(str(para) for para in matching_arg[5:]) 
    
            matching_type,sub_folder,weight,cores_for_sampling, cores_for_obj = self.get_objective_input(matching_arg) 
            
            self.print_initialization_properties_matching(matching_type,
                                                     sub_folder,
                                                     weight,
                                                     cores_for_sampling, 
                                                     cores_for_obj,
                                                     matching_argument) 

            total_cores_by_sampling += cores_for_sampling 
        
            total_cores_by_obj += cores_for_obj

            type_subfolder.append( matching_type + " " + sub_folder ) 

            # ----------- set up the working directory ----------------
            self.transfer_prepsystem_to_working_folders(matching_type,sub_folder) 

            ref_address = os.path.join(self.reference_folder_address,matching_type,sub_folder) 

            IO.check_dir.decide_folder_status(ref_address,"Reference data") 

            predict_address = os.path.join(self.predicted_folder,matching_type,sub_folder) 
    
            # get folder paths of predicted folder 
            num_predict_folders,predict_sub_folders = self.Get_Path(predict_address)  

            # get folder paths of all reference folders  
            num_ref_folders,ref_sub_folders = self.Get_Path(ref_address)  

            # sort and make the folder list as tuples 
           
            predict_sub_folders = tuple(sorted(predict_sub_folders)) 

            ref_sub_folders = tuple(sorted(ref_sub_folders)) 

            # ---------- Initialize sampling method ------------
            
            self.sampling_method = sampling.MD.run_as_subprocess(self.packagename,
                                         matching_type,
                                         predict_sub_folders,
                                         num_predict_folders,
                                         self.command_lst[0],
                                         cores_for_sampling,
                                         self.HOME) 

            # ---------- Initialize output path and argument  ----

            self.ref_address_dict[matching_type] = {sub_folder: ref_sub_folders} 
        
            self.predict_address_dict[matching_type] = {sub_folder: predict_sub_folders} 

            self.argument_dict[matching_type] = {sub_folder: 
                                                 ( weight,
                                                   cores_for_sampling, 
                                                   cores_for_obj,
                                                   matching_argument)} 

        self.check_all_cores(total_cores,total_cores_by_sampling,total_cores_by_obj )

        self.dectect_duplicated_folder(type_subfolder) 

        return None  

    def finish(self):  

        return self.ref_address_dict, self.predict_address_dict, self.argument_dict, self.sampling_method,self.lastline

