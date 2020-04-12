# Python standard library:
import subprocess   
import sys
import time
import logging
import os 
# customized library:
import sampling.potential_LAMMPS 

# Launch and Terminate sampling jobs 

class run_as_subprocess(): 

    """Invoke external sampling softwares of choice to calculate properties using force-field  
    potential in every iterations:  
    
    -Parameters: 
    ------------

    matching_type: 
    e.g."rdf", "force" ....  

    wk_folder: which folders simulations should be run 
    e.g. job_1234/predicted/rdf/ 

    cores: number of cores used in simulation
    e.g. srun -n 4  or mpirun -np 4  

    -Return: 
    --------

    finish: True if all jobs run successfully and False otherwise   
    """ 

    # Class variables visible for all objects

    command_list_cls = [ ] 
    
    wk_folder_list_cls = [ ] 

    @classmethod
    def __init__(cls,
                 packagename,
                 matching_type,
                 wk_folder_lst,
                 num_jobs,
                 command,
                 total_cores_assigned,
                 HOME): 

        cls.packagename= packagename

        cls.matching_type = matching_type 


        cls.sampling_cores_assignment(matching_type,
                                      num_jobs,
                                      total_cores_assigned)
        cls.num_jobs = num_jobs 

        command_modified = command%(cls.cores_per_job,matching_type) 

        cls.command = command_modified  

        # For every matching type, save their command and working folders into list 

        cls.update_matching(command_modified,wk_folder_lst) 
        
        # Print the Initialization information:  

        cls.print_initialization() 

        cls.HOME = HOME 

        return None  

    @classmethod 
    def sampling_cores_assignment(cls,matching_type,num_jobs,total_cores_assigned): 

        cores_logger = logging.getLogger(__name__) 

        cls.cores_per_job = int(total_cores_assigned/num_jobs) 
        
        if ( cls.cores_per_job == 0): 

            cores_logger.error("ERROR: "+ matching_type  + " : " 
                            + "The total number of cores requested ( %d ) through input file "%total_cores_assigned
                            + "is less than the number jobs ( %d ) ...\n"%(num_jobs)
                            + "Performance degration due to overutilized cores ! " )
    
                        

            sys.exit("Check errors in the log file")

        return None  

    @classmethod
    def print_initialization(cls):  

        logger = logging.getLogger(__name__)

        logger.info("------------------------- Initialize sampling method: %s -------------------------\n\n"%(cls.packagename)) 
        
        logger.info("Number of jobs: %d \n"%cls.num_jobs) 
    
        logger.info("Number of cores used per job:  %d \n"%( cls.cores_per_job )) 

        logger.info("Command:  %s \n",cls.command) 

        return None 

    @classmethod    
    def Launch_Jobs(cls,cmd,joblist): 

        out = open("output","w") ; error = open("error","w") 

        joblist.append( subprocess.Popen(cmd,\

            stdout=out, stderr=error,shell=True) )  

        return joblist  

    @classmethod
    def update_matching(cls,command,working_folders):

        cls.command_list_cls.append(command)

        cls.wk_folder_list_cls.append(working_folders) 

        return None   

    @classmethod
    def run(cls,type_name,force_field_parameters): 

        """
        
        Iteratively change into each working folder to run 
        simulations from commandline.   

        -Parameters: 
        ------------ 

        Working folders list ( wk_folder_list_cls): 
        change to target folders corresponding to properties 

        Command list ( command_list_cls): 
        apply command corresponding to proeprties   

        -Return: 
        --------

        exit_code for each subprocess after they finish  

        """ 

        run_logger = logging.getLogger(__name__) 

        run_logger.debug( "Ready to Run jobs ... " ) 

        #Use_LAMMPS_Potential(cls.potential_type,cls.wk_folder_list_cls,force_field_parameters)             

        output_content_dict = sampling.potential_LAMMPS.choose_lammps_potential(type_name,force_field_parameters)  
        
        sampling.potential_LAMMPS.propagate_force_field(cls.wk_folder_list_cls,output_content_dict)     
        
        All_jobs = []  

        # iterate each type of properties in the list 

        for indx,cmd in enumerate(cls.command_list_cls):  

            each_matching_type  = []  

            for folder in cls.wk_folder_list_cls[indx]:  

                os.chdir(folder)    
                
                each_matching_type = cls.Launch_Jobs(cmd,each_matching_type)            

                time.sleep(0.02) 

            All_jobs.append(each_matching_type) 

        cls.exit_codes = [ [ job.wait() for job in matching_type ] for matching_type in All_jobs ]  
        
        os.chdir(cls.HOME)  

        run_logger.debug("All LAMMPS jobs are launched ... ") 

        return None 

    @classmethod 
    def exit(cls):  

        """
        
        Check if LAMMPS jobs ran successfully: 
        

        -Parameters: 
        ------------

        exit_code "0" means job is sumbitted and run successfully 
        error file size = 0 means no error messages written 

        -Return: 
        --------

        finish = True if all jobs exit successful.  

        """
    
        exit_logger = logging.getLogger(__name__) 

        for type_index,matching in enumerate(cls.wk_folder_list_cls):

            for sub_index,fd in enumerate(matching): 

                if (  os.stat(fd+"/error").st_size == 0 

                    and cls.exit_codes[type_index][sub_index]==0):  

                    continue 

                else: 

                    error_command = cls.command_list_cls[type_index] 
    
                    at_folder = cls.wk_folder_list_cls[type_index][sub_index]

                    exit_logger.error( "ERROR: Command: %s, Folders: %s "\
                                        %( error_command, at_folder)) 

                    return False

        exit_logger.debug("LAMMPS exits successfully") 

        return True 
                    

