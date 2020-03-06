# Python standard library:
import subprocess   
import sys
import time
import logging
import os 
# customized library:
import potential 

# Wrapper functions to invoke LAMMPS as a MD engines 

def Invoke_LAMMPS(in_para,predict_address,TOTAL_CORES_ASSIGN,HOME): 

	lammpslogger = logging.getLogger(__name__) 

	lammpslogger.debug("Initialize LAMMPS ...") 
	
	total_cores_requested = 0

	# loop over all type of property matching 
	for matching in in_para.matching:  		
				
		list_arg = matching.split()

		typename = list_arg[0] 

		# get the number of cores requested by user  

		cores_requested = int(list_arg[3])  
	
		subfolder = list_arg[1] 
	
		wk_folders_lst = predict_address[typename][subfolder] 

		num_jobs = len(wk_folders_lst) 

		cores_per_job = Core_Assigment(typename,num_jobs,cores_requested)

		command = in_para.run_command 

		ptype = in_para.ptype

		command = command%(cores_per_job,typename) 	

		LAMMPS = RunSimulations("LAMMPS",matching,wk_folders_lst,command,cores_per_job,HOME,ptype=ptype) 
		
		total_cores_requested = total_cores_requested + cores_requested 

	if ( total_cores_requested > TOTAL_CORES_ASSIGN ):  

		lammpslogger.error("ERROR:Total cores requested for all LAMMPS jobs are more than assigned by Slurm ") 

		sys.exit() 

	else: 

		lammpslogger.debug("LAMMPS initialization finish ...") 

		return LAMMPS 

# Determine and check the number of cores used by each job 
def Core_Assigment(matchingtype,num_jobs,total_cores_assigned): 

	runlogger = logging.getLogger("Core_Assignment: ") 	

	cores_per_job = total_cores_assigned/num_jobs	

	if ( cores_per_job ==0): 

		cores_per_job = 1 

		runlogger.warning("WARNING: "+ matchingtype  + " : " 
						+ "The number of cores assigned, %d "%(total_cores_assigned)
						+ "is less than the number jobs, %d  ... "%(num_jobs) )

	return cores_per_job 

# Launch and Terminate LAMMPS jobs 
class RunSimulations(): 

	""" 
	Invoke simulations engines to calculate properties using  force-field    
	potential in every iterations.  
	
	-Parameters: 
	------------

	matchingtype: 
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

	Name = "In simulation_engine_mod " 

	@classmethod 
	def __init__( 

			cls,package,matchingtype,wk_folder_lst,command,cores,HOME,ptype=None): 

		if ( ptype ): cls.potential_type = ptype

		cls.package= package

		cls.matchingtype = matchingtype 

		cls.command = command

		cls.cores = cores

		cls.num_jobs = len(wk_folder_lst) 

		cls.HOME = HOME 

		# For every matching type, save their command and working folders into list 

		cls.Update_Matching(command,wk_folder_lst) 
		
		# Print the Initialization information:  

		cls.Print_Initialization() 

		return None  
	
	@classmethod
	def Print_Initialization(cls):  

		runlogger = logging.getLogger(__name__) 
	
		runlogger.info("------------------------- Initialize %s for %s matching-------------------------\n"%(cls.package,cls.matchingtype)) 

		runlogger.info("Potential type: %s \n"%cls.potential_type) 

		runlogger.info("Number of jobs: %d \n"%cls.num_jobs) 
	
		runlogger.info("Number of cores used per job:  %d \n"%( cls.cores )) 

		runlogger.info("Command:  %s \n", cls.command) 

		return None 

	@classmethod
	def Update_Matching(cls,command,working_folders):

		cls.command_list_cls.append(command)

		cls.wk_folder_list_cls.append(working_folders) 

		return None   

	@classmethod	
	def Launch_Jobs(self,cmd,joblist): 

		out = open("output","w") ; error = open("error","w") 

		joblist.append( subprocess.Popen(cmd,\

			stdout=out, stderr=error,shell=True) ) 	

		return joblist  

	@classmethod
	def Run(cls,force_field_parameters): 

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

		runlogger = logging.getLogger(cls.Name+ "Run: ") 

		runlogger.debug( "Ready to Run jobs ... " ) 

		#Use_LAMMPS_Potential(cls.potential_type,cls.wk_folder_list_cls,force_field_parameters) 			

		output_content_dict = potential.choose_lammps_potential(cls.potential_type,force_field_parameters)	
		
		potential.propagate_force_field(cls.wk_folder_list_cls,output_content_dict) 	
		
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

		runlogger.debug("All LAMMPS jobs are launched ... ") 

		return None 

	@classmethod 
	def Exit(cls):  

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
	
		runlogger = logging.getLogger(cls.Name + "Exit: ") 
		
		for type_index,matching in enumerate(cls.wk_folder_list_cls):

			for sub_index,fd in enumerate(matching): 

				if (  os.stat(fd+"/error").st_size == 0 

					and cls.exit_codes[type_index][sub_index]==0):  

					continue 

				else: 

					error_command = cls.command_list_cls[type_index] 
	
					at_folder = cls.wk_folder_list_cls[type_index][sub_index]

					runlogger.error( "ERROR: Command: %s, Folders: %s "\
										%( error_command, at_folder)) 

					return False

		runlogger.debug("LAMMPS exits successfully") 

		return True 
					

