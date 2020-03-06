import numpy as np 
import os 
import logging
import time 
import subprocess 
import argparse
import const_mod 
import sys 
import simulation_engine_mod as simulation  

def is_int(a): 

	try: 

		int(a) 	

	except ValueError: 

		return False 

	return True 	

def is_string(a):  

	try: 

		str(a) 

	except ValueError: 

		return False 

	return True 

def is_float(a): 

	try: 

		float(a) 

	except ValueError: 

		return False 

	return True 

def get_lines(filename):

	with open(filename,"r") as content:

		for i,l in enumerate(content):

			pass

	return i+1

#-----------------------------------------------------------------------------
#---------------------------- Command Line -----------------------------------
#-----------------------------------------------------------------------------

class ReadCommandLine():

	@classmethod
	def __init__(cls): 

		return None  

	@classmethod
	def Finish(cls):  

		cls.Take_Command_Line_Args() 

		cls.Parse_Input_Argument()
	
		cls.logger = cls.Set_Run_Mode("optimization_" + cls.JOBID +".log",cls.MODE)  
			
		return cls.logger,cls.TOTAL_CORES,cls.INPUT, cls.JOBID

	@classmethod 
	def Take_Command_Line_Args(cls): 
		
		parser = argparse.ArgumentParser()

		parser.add_argument("-c", "--cores", type=int, required=True)

		parser.add_argument("-i", "--input", type=str, required=True)

		parser.add_argument("-j", "--job", type=str, required=True) 
		
		#parser.add_argument("-n", "--node", type=str, required=True)

		parser.add_argument("-m", "--mode", type=str, required=False)

		args = parser.parse_args()

		cls.argument = dict( args.__dict__.iteritems() )  

		return None  

	@classmethod
	def Parse_Input_Argument(cls):  

		cls.JOBID = cls.argument["job"]  
	
		cls.TOTAL_CORES = cls.argument["cores"] 

		if ( cls.argument["mode"] == None):

			cls.MODE = "run" 

		else: 

			cls.MODE = "debug" 

		cls.INPUT = cls.argument["input"] 

		return None  

	@classmethod
	def Select_Run_Mode(cls,arg): 

		mode = { 
		
		"debug": logging.DEBUG, 
		"run": logging.INFO

		}  
		
		mode.get(arg,"invalid") 

		if ( mode[arg] == "invalid"):  

			print "Invalid Run mode: Choose debug or run"

			sys.exit()  

		return mode[arg] 

	@classmethod
	def Select_Formatter(cls,arg): 

		mode = { 
		
		"debug": "%(asctime)s - %(name)s - %(levelname)s - %(message)s", 
		"run": "%(message)s" 
		}  

		mode.get(arg,"invalid") 
		
		if ( mode[arg] == "invalid"): 

			print "Invalid Run mode: Choose debug or run"

			sys.exit()  
		
		return mode[arg] 
		
	@classmethod
	def Set_Run_Mode(cls,logname,mode): 

		logger = logging.getLogger() 

		logger.setLevel(cls.Select_Run_Mode(mode)) 

		fh = logging.FileHandler(logname,mode="w")

		formatter = logging.Formatter(cls.Select_Formatter(mode)) 

		fh.setFormatter(formatter) 

		logger.addHandler(fh) 

		return logger

	def choose_cores_for_LAMMPS(self,total_atoms,max_cores): 
		
		"""
		Determine how many cores needed to run LAMMPS. 

		--Argument: 

		total_atoms: system size  

		max_cores: max number of cores requested

		--Returns: 

		num_cores: number of cores assigned to LAMMPS 

		"""

		if ( max_cores <= 8 ): 

			return max_cores 	

		elif ( 10 <= max_cores ): 

			if ( total_atoms <= 500 ):  

				return min(max_cores,4) 

			elif ( 500 < total_atoms <= 1000) : 

				return min(max_cores,6)    	

			elif ( 1000 < total_atoms <= 2000 ) : 

				return min(max_cores,10) 	
		
		elif ( 2000 < total_atoms <= 3000 ) : 
		
			return min(max_cores,14)   

		elif ( 3000 < total_atoms <= 4000 ) :  

			return min(max_cores,18)  

		elif ( 4000 < total_atoms <= 5000 ) : 

			return min(max_cores,22)  

#-----------------------------------------------------------------------------
#---------------------------- Input Parameters -------------------------------
#-----------------------------------------------------------------------------

class ParseInputFile():  
	# each line of data must matched with the order of following inputkeyword

	inputkeyword = ["units",
					"matchingtype",
					"command",
					"restart",
					"guess_parameter",
					"fit_and_fix",
					"constraints",
					"options",
					"argument"] 

	def __init__(self,filename): 

		# get the file name

		self.filename = filename 

		# number of lines to pointer for certain keywords 	
		
		self.pointer = 0 
		
		try: 

			with open(self.filename,"r") as inputfile: 

				pass

		# except FileNotFoundError #in Python 3

		except IOError:  

			print "Input file for optimizations is not found"

			# terminate the program 

			sys.exit() 

		self.Read_Content()
	
		self.Get_Parameters() 	

		return None 

	# Read all contents from input file into a variable "self.parameters"  

	def Read_Content(self):

		with open(self.filename,"r") as inputfile: 
		
			self.parameters = []  

			for line in inputfile:

				contents = line.split()    
			
				# skip the empty lines 

				if ( contents == [] ): 

					continue 

				# skip the comment due to "#" or "&" 

				elif ( "#" in contents[0] or "&" in contents[0]): 

					continue

				else:

					self.parameters.append(contents) 

		return None 
		
	# Error Checking for each type of input  

	def CheckJobname(self): 

		if ( len(self.parameters[self.pointer]) > 1): 

			print "Only 1 job name should be provided; more than 1 is given here "

			sys.exit()  			
		
		return None 	

	def CheckRestart(self): 

		if ( len(self.parameters[self.pointer]) > 1): 

			print "Only 1 restart frequency value should be provided; more than 1 is given here "

			sys.exit()  			
		
		return None 
	
	def CheckGuess(self): 

		if (self.guess_parameter.size != self.fit_and_fix.size ): 

			print "The number of parameters provided is not equal to fitted parameters (=1) + unfitted parameters (=0) " 

			sys.exit()  

	def CheckFitFix(self): 

		if ( np.any(self.fit_and_fix > 1 ) or np.any(self.fit_and_fix < 0 )) : 

			print "Fit Fix ERROR: only 1 or 0 is allowed "

	def CheckMode(self):

		if ( self.mode != "Restart" and  self.mode != "Perturb"):  

			print "MODE ERROR! : Check the spelling or type: must choose either 'Restart' or 'Perturb' to initialize Nelder-Mead Simplex  "

			sys.exit() 

	def CheckPerturb(self): 

		fit_size = ( self.fit_and_fix==1 ).sum()  

		if ( self.mode == "Perturb" and fit_size != self.perturb.size ):   	

			print  "Simplex Size ERROR: The number of fitted parameters (=1) should be equal to number of perturbed parameters " 

			sys.exit() 

		if ( self.mode =="Restart" and fit_size != self.purturb.size -1 ): 

			print "Simplex Size ERROR: The number of fitted parameters + 1  (=1) should be equal to number of vertices " 

			sys.exit() 

	def CheckRestartpara(self): 

		vertices_shape = self.vertices.shape  
	
		if ( vertices_shape[0] <= vertices_shape[1] ): 

			print "The number of vertices is larger than the number of objective functions"	

			sys.exit() 

		if ( vertices_shape[0] != self.obj.size ): 

			print "Restart Simplex ERROR!: number of vertices are not equal to number of objectives" 

			sys.exit() 

	def CheckConstraints(self):  

		if ( len(self.constraints )%3 != 0 or len(self.constraints ) < 3 ): 
		
			print "Constraints ERROR: index, lowerbound, upperbound should be provided together" 

			sys.exit() 

		if ( np.amax(self.constraints_index) > self.guess_parameter.size -1 ): 

			print "Constraints ERROR: The constraints index provided is out of bound" 

			sys.exit() 

		for cindex in self.constraints_index:

			if ( cindex in self.unfit):

				print "Constrains ERROR: Constraint index has to be fitted variable (=1). Unfitted variable (=0) can not be constrained "

				sys.exit()
				 
	# define functions to read contents

	def Seed(self,seedval): 

		try: 

			self.seed = int(seedval) 
			
		except ( ValueError, TypeError): 

			print "Input file: seed value must be an integer value " 

			sys.exit() 

		return None 

	def units(self):  

		units = self.parameters[self.pointer]

		self.UNITS = const_mod.Units(str(units[0])) 		
		
		self.pointer = self.pointer + 1 

		return None 
	
	def This_line_is_matchingtype(self,a):  

		results = [ is_string(a[0]), is_int(a[1]), is_float(a[2]) , is_int(a[3]), is_int(a[4]) ] 

		num_True = results.count(True) 

		return num_True 

	def matchingtype(self): 	

		self.matching = [] 
		
		counter = 0 

		while True:

			parse_matching_para = self.This_line_is_matchingtype(self.parameters[self.pointer]) 

			if ( parse_matching_para == 5):   

				matching_type = " ".join(self.parameters[self.pointer]) 
	
				self.matching.append(matching_type)  

				self.pointer = self.pointer + 1 

				counter = counter + 1  

			elif ( 3 <= parse_matching_para < 5 ): 
	
				print "Format ERROR: Make Sure your input follows: matchingtype (str), ID (int) , weight (float), cores for MD (int), cores for analysis (int) "

				sys.exit() 
		
			else: 	

				break 

		return None 

	def command(self):

		run = " " 
	
		run_command = run.join(self.parameters[self.pointer])

		self.run_command = run_command  

		self.pointer = self.pointer + 1 
		
		return None 

	# get the restart frequency: 

	def restart(self): 

		try: 
	
			self.restart = int(self.parameters[self.pointer][0]) 
			
		except ( ValueError, TypeError): 

			print "Input file: Restart frquency must be an integer value "

			sys.exit() 

		self.CheckRestart()

		self.pointer = self.pointer + 1 

		return None 

		# get the guess parameters:  

	def guess_parameter(self):  

		try: 

			self.ptype = self.parameters[self.pointer][0]

			self.guess_parameter = np.array(self.parameters[self.pointer][1:]).astype(np.float64) 

		except ( ValueError, TypeError): 

			print "Input file: Guess Parameters ERROR!: can't read guess data ! Make sure they are all floats  "

			sys.exit() 

		if ( is_float(self.ptype) or is_int(self.ptype) ): 
			
			print "The first guess parameters must be potential type (string) " 
			
			sys.exit()    

		self.pointer = self.pointer + 1 

		return None 

	
	# get the fitted or unfitted parameters: 

	def fit_and_fix(self):  

		try: 

			self.fit_and_fix = np.array(self.parameters[self.pointer]).astype(np.int) 

		except ( TypeError,ValueError): 

			print "Input file: fit and fix ERROR!: can't read fit and fix data ! Make sure they are all integers "
	
			sys.exit()  

		self.fit = np.array([ i for i,x in enumerate(self.fit_and_fix) if x == 1  ],dtype=np.int) 

		self.unfit = np.array([ i for i,x in enumerate(self.fit_and_fix) if x ==0 ],dtype=np.int) 

		self.guess = self.guess_parameter[self.fit] 
	
		self.CheckGuess() 

		self.CheckFitFix() 

		self.pointer = self.pointer + 1 

		return None 
	
	# get the constraints 

	def constraints(self):  

		indx = self.inputkeyword.index("constraints") 				

		self.constraints = np.array(self.parameters[self.pointer])

		num_constraints = len(self.constraints)/3 

		try: 
			self.constraints_index = np.array([self.constraints[idx*3]  for idx in xrange(num_constraints)]).astype(np.int)-1 

			self.old_inddex = self.constraints_index + 1  

			for nindex in xrange(self.constraints_index.size): 

				num_shift = sum( i < self.constraints_index[nindex] for i in self.unfit) 

				self.constraints_index[nindex] = self.constraints_index[nindex] - num_shift 

			self.constraints_bound = np.array([ [ self.constraints[3*indx+1], self.constraints[3*indx+2]]   for indx in xrange(num_constraints)]) 	

		except ( ValueError, TypeError) : 

			print "Input file: Constraints ERROR!: can't read constraints parameter ! Make sure their types are correct " 

			sys.exit() 

		self.CheckConstraints() 

		self.pointer = self.pointer + 1 

		return None

	# get the options of either Perturb or Restart  

	def options(self): 
	
		if ( "options" in self.inputkeyword ):

			#indx = self.inputkeyword.index("options") 
		
			try: 
			
				self.mode = self.parameters[self.pointer][0]  
				
			except ( ValueError, TypeError): 

				print "Input file: Keyword option ERROR!: Specify Restart or Perturb"

				sys.exit() 
					
			self.CheckMode() 

			self.pointer = self.pointer + 1

		return None 	

	# get the argument of either perturb or restart
		
	def argument(self):  

		#indx = self.inputkeyword.index("argument") 

		if ( self.mode =="Perturb"):  

			self.seed = None 

			arglist = self.parameters[self.pointer-1:self.pointer]
			
			if ( len(arglist[0]) == 2 ): 

				self.Seed(arglist[0][1]) 

			elif ( len(arglist[0]) > 2 ): 

				print "Too many arguments for Perturb: at most 2 are needed " 
	
				sys.exit() 

			try: 

				self.perturb = np.array(self.parameters[self.pointer:self.pointer+1][0] ).astype(np.float64)  

				self.pointer = self.pointer + 1 

			except ( ValueError, TypeError): 

				print "Input file: Perturb parameters ERROR! They should be all floats"
				
				sys.exit() 

			self.CheckPerturb() 

		elif ( self.mode =="Restart"): 

			try: 

				self.obj = np.array(self.parameters[self.pointer:self.pointer+1][0] ).astype(np.float64)
				
				num_v = self.obj.size   

				self.pointer = self.pointer + 1 
				
				self.vertices = np.array(self.parameters[self.pointer:self.pointer+num_v+1] ).astype(np.float64) 
					
				self.pointer = self.pointer + num_v 	
					
			except ( ValueError, TypeError) : 

				print "Input file: Restart parameters ERROR! They should be all floats"  

				sys.exit() 	

			self.CheckRestartpara() 

		return None 

	def PrintInput(self): 

		num_fit = (self.fit_and_fix==1).sum()  
		
		num_fix = (self.fit_and_fix==0).sum()

		unfit = np.array([ i for i,x in enumerate(self.fit_and_fix) if x ==0 ],dtype=np.int)

		fit = np.array([ i for i,x in enumerate(self.fit_and_fix) if x == 1  ],dtype=np.int)

		print "The number of Vertices: ", fit.size + 1  
		print "\n" 
		print "The input parameters are:  "
		print "\n" 
		print self.guess_parameter.tolist()  
		print "\n"
		print "The %s fitting parameters: " %(num_fit) 
		print "\n"
		print (fit+1).tolist()   
		print "\n "
		print "The %s fixed parameters: "%(num_fix) 
		print "\n"
		print (unfit+1).tolist()  
		print "\n"
		print "The %s constrained parameters: " %(self.constraints_index.size) 
		print "\n"	
		print self.old_inddex.tolist()  

		return None 	

	def Get_Parameters(self):

		for para in self.inputkeyword: 
	
			each_para = getattr(self,para) 

			each_para() 

		return None 

#-----------------------------------------------------------------------------
#---------------------------- Folders Setup -----------------------------
#-----------------------------------------------------------------------------

class Setup(): 

	ref_folder = "../ReferenceData"

	prep_folder = "../src/prepsystem"

	predict_folder = "Predicted"

	def __init__(self,jobid,matching_in): 

		self.home = os.getcwd() 

		self.matching_type = np.unique([ everytype.split()[0]\
							 for everytype in matching_in ])	

		self.subfolders_per_type = self.Get_subfolders_from_type(self.matching_type,matching_in )		

		self.jobid = jobid 

		self.Check_Folders(self.ref_folder,self.matching_type,self.subfolders_per_type)

		self.Check_Folders(self.prep_folder,self.matching_type,self.subfolders_per_type)

		self.Mkdir() 

		return None 

	def Get_subfolders_from_type(self,matching_type,matching_in ): 

		subfolders_per_type = {} 
	
		for typename in matching_type: 	

			type_sub = [] 
	
			for everytype in matching_in: 

				args = everytype.split() 

				if ( args[0] == typename): 

					type_sub.append(args[1]) 
				
			subfolders_per_type[typename] = type_sub	
		
		return subfolders_per_type 

	def Check_Folders(self,address,matching_type,subfolders_per_type):

		folder_exists = os.path.isdir(address) 

		if ( not folder_exists ): 

			print "ERROR: folder " + address + " does not exist" 

			sys.exit()

		if ( len(os.listdir(address+"/" )) == 0 )  : 

			print "ERROR: No folders inside " + address   
		
			sys.exit() 

		for match in matching_type: 

			for subfolder in subfolders_per_type[match]: 

				match_folder = address + "/" + match + "/" + subfolder 
				
				if ( not os.path.isdir( match_folder)): 

					print "ERROR: folders: " + match_folder 
					print "Check the following: \n"
					print "1. Does the type %s exist ?"%match 
					print "2. Is this folder name consistent with the matching type name used in input file e.g. isobar, force etc ?"
					print "3. Does the subfolder (e.g. 1 2 ...) index of this type exist ?"
		
					sys.exit()	

				if ( len(os.listdir(match_folder+"/")) ==0):
			
					print "ERROR: No files inside: ",match_folder 		
		
					sys.exit() 
		
		return None

	def Mkdir(self): 	
		
		job_folder = "Job_" + self.jobid + "/"

		working_folders = job_folder + self.predict_folder 

		os.mkdir(job_folder) 
		
		os.mkdir(working_folders)

		os.mkdir(job_folder + "Restart") 

		os.mkdir(job_folder + "Output")

		self.predict_folder_list= [ ] 

		self.ref_folder_list = [] 

		for everytype in self.matching_type: 
	
			self.job_folders = working_folders + "/" + everytype + "/" 
	
			os.mkdir(self.job_folders)	

			self.predict_folder_list.append(self.job_folders)  
		
			self.ref_folder_list.append(self.ref_folder + "/" + everytype ) 
	
		return None 

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
	
	def Get_Path(self,wk_folder): 

		os.chdir(wk_folder) 

		sub_folders = []  

		sub_folders = self.Go_to_Subfolders(sub_folders) 

		num_folders = len(sub_folders)  	

		os.chdir(self.home) 

		return num_folders,sub_folders 

	def Get_Folders_Path(self,main_folders,sub_folder_type):  

		job_address = { } 

		for indx,folder in enumerate(main_folders): 

			sub_index_type = {} 
							
			for sub_indx in sub_folder_type[self.matching_type[indx]]: 	

				num_jobs,sub_folders = self.Get_Path(folder + "/" + sub_indx) 

				sub_index_type[sub_indx] = sub_folders  
	
				job_address[self.matching_type[indx]] = sub_index_type 

		return job_address  

	def Transfer_Prepsystem_To_Predict_Folder(self): 

		for indx,matching_type in enumerate(self.matching_type): 

			for sub_indx in self.subfolders_per_type[matching_type]:  

				subprocess.call("cp"+ " -r " 
								+ self.prep_folder + "/" + "%s/%s" %( matching_type,sub_indx )
								+  " " + self.predict_folder_list[indx], shell=True ) 

		return None 

	def Finish(self): 

		self.Transfer_Prepsystem_To_Predict_Folder()

		ref_job_address = self.Get_Folders_Path( self.ref_folder_list,self.subfolders_per_type ) 	

		predict_job_address = self.Get_Folders_Path( self.predict_folder_list,self.subfolders_per_type )

		return self.home,ref_job_address,predict_job_address

#-----------------------------------------------------------------------------
#---------------------------- Output -----------------------------------------
#-----------------------------------------------------------------------------

class Output(): 

	def __init__(self,input_para,jobID): 

		self.jobID = jobID 

		self.restart_address = "Job_" + str(jobID) + "/" + "Restart" + "/" 

		self.output_address  = "Job_" + str(self.jobID) + "/" + "Output" + "/" 

		self.input_para = input_para

		self.restart_log_file = self.restart_address + "log.restart"
	
		self.current_restart_file = self.restart_address + "current.restart"	

		self.vertices_file = self.output_address + "best_worst.vertices"

		self.matching_func_file = self.output_address + "costfunction.txt"
		
		open(self.restart_log_file,"w").close()
	
		open(self.current_restart_file,"w").close()
		
		open(self.vertices_file,"w").close() 

		open(self.matching_func_file,"w").close() 
		
		return None 

	def Write_Cost_Func(self,itera,costfunction): 

		with open(self.matching_func_file,"a") as out: 

			np.savetxt(out,zip(itera,costfunction)) 	

		return None 	

	def Write_Best_Worst_Vertices(self,itera,best_vertices,worst_vertices):
	
		with open(self.vertices_file,"a") as out:

			out.write(zip(itera,best_vertices,worst_vertices))
						
		return None 

	def Write_Restart(self,itera,best_parameter,simplex):

		if ( itera % self.input_para.restartfreq == 0 ): 

			best_and_fix = np.zeros(self.input_para.fit_and_fix.size) 

			best_and_fix[self.input_para.fit_and_fix==1] = best_parameter 

			best_and_fix[self.input_para.fit_and_fix==0] = self.input_para.guess_parameter[self.input_para.fit_and_fix == 0]   

			self.best_and_fix = best_and_fix 

			with open(self.restart_log_file,"a") as f: 
			
				self.RestartContent(f,itera,simplex)

			with open(self.current_restart_file,"w") as f: 

				self.RestartContent(f,itera,simplex)

		return None 

	def RestartContent(self,f,itera,simplex):  

		contents_header = "# Iteration: " + str(itera) + "\n"

		bestparameter_header = "# Guess initial Parameters: \n"

		f.write(contents_header) 

		f.write("\n")

		for match in self.input_para.matching: 
	
			f.write(match + "\n") 	

		f.write(self.input_para.run_command + "\n") 

		f.write(self.input_para.restart) 

		#f.write(self.best_and_fix) 

		f.write("\n")

		f.write(bestparameter_header) 		

		np.savetxt(f,self.best_and_fix,newline=" ",fmt="%.10f") 

		f.write("\n") 
		f.write("\n") 

		f.write("# fit (1) and fix (0) parameters: ") 

		f.write("\n") 
		f.write("\n") 

		np.savetxt(f,self.input_para.fit_and_fix,newline=" ",fmt="%d") 
		
		f.write("\n") 
		f.write("\n") 

		f.write("# constraints: ")

		f.write("\n") 
		f.write("\n") 

		np.savetxt(f,self.input_para.constraints,newline=" ", fmt="%s") 		

		f.write("\n") 
		f.write("\n") 

		f.write("# create (Perturb) or use existing vertices (Restart): ") 
		f.write("\n") 

		f.write("Restart") 
		f.write("\n" ) 

		np.savetxt(f,simplex.cost,newline=" ",fmt="%.8e") 
		f.write("\n") 
		f.write("\n") 
	
		np.savetxt(f,simplex.vertices,fmt="%.10f") 

		f.write("\n") 
		f.write("\n")
		
		return None 

def Compute_N_match(T_predict_sort,T_ref_sort):  

	N_match = 0 

	for N in zip(T_predict_sort,T_ref_sort): 

		if ( N[0] == N[1]): 

			N_match = N_match + 1 

	return N_match 

def Penalty( N_tol,N,T_predict_sort,T_ref_sort): 

	N = T_ref_sort.size 

	N_match = Compute_N_match(T_predict_sort,T_ref_sort) 

	penalty = max(N - N_tol - N_match, 0 )  

	return penalty 

def Initialize_Isobar_Matching(in_para): 

	counter = 0 

	T = {} 
	
	ref_density = {} 

	ref_density_norm = {} 

	N_tol = {} 

	for match in in_para.matching: 

		arg = match.split() 	
		
		if ( arg[0] == "isobar"): 

			T_start = arg.index("T") + 1  

			T_end = arg.index("sort") 

			counter = counter + 1 

			T[str(counter)] = np.array(arg[T_start:T_end]).astype(np.int)   
		
			N_tol[str(counter)] = int(arg[-1]) 	

	return N_tol,T

def Compute_Isobar_Norm(ref_address,natoms): 

	ref_file = ref_address["isobar"]

	ref_density = {}  

	ref_density_norm = {} 

	T_rank_ref = {} 

	for subfolder in ref_file: 
	
		list_of_address = ref_file[subfolder] 

		density_all_T = np.zeros(len(list_of_address))  

		for indx,address in enumerate(list_of_address): 
		
			data = np.loadtxt(address + "/Ref.density" )
		
			volume = np.average(data[:,0]) 

			density = (natoms/const_mod.NA)*const_mod.water_mol_mass/(volume*UNITS.vol_scale*10**6) 			
	
			density_all_T[indx] = density
	
		index = np.argsort(density_all_T) 
	
		T_rank_ref[subfolder] = T[subfolder][index]

		ref_density[subfolder] = density_all_T 

		ref_density_norm[subfolder] = np.var(ref_density[subfolder]) 
		
	return ref_density, ref_density_norm,T_rank_ref 

def Compute_Density(natoms,sub_folders,num_T,real_units):  	

	volume = np.zeros(num_T) 

	for indx,address in enumerate(sub_folders):

		volume[indx] = np.average( np.loadtxt(address+"/dump.volume" ) ) 	

	return  ( natoms/const_mod.NA)*const_mod.water_mol_mass/(volume*real_units.vol_scale*10**6)  

def Isobar_Matching(natoms,file_address,UNITS):

	costfunc = 0 

	for subfolder in file_address: 

		list_address = file_address[subfolder] 

		num_T = len(list_address) 
	
		predict_density = Compute_Density(natoms,list_address,num_T,UNITS) 	

		indx = np.argsort(predict_density) 

		T_predict_sort = np.array(T[subfolder]).astype(np.int)[indx] 

		diff = predict_density - ref_density[subfolder]

		penalty = Penalty( N_tol_all[subfolder],num_T,T_predict_sort,T_rank_ref[subfolder]) 

		scaled_isobar = np.sum((diff)**2/ref_density_norms[subfolder]) 

		print predict_density 
		print ref_density[subfolder] 
		print scaled_isobar  
		print penalty
		print scaled_isobar 

		costfunc = costfunc + scaled_isobar*penalty + scaled_isobar
			
	return costfunc

def ComputeObjective_by_Lammps(w,force_field_parameters):

	costfunclogger = logging.getLogger("In Cost function evalutations: ") 

	LAMMPS.Run(force_field_parameters)

	job_runs_successful = LAMMPS.Exit() 

	time.sleep(5) 

	costfunclogger.debug("Start Reading output from LAMMPS ... ") 	

	#outputfinsih = LammpsOutputReady(outputfiles)

	#not_finishing_rdf_traj(dcdfile,ref_configs=2001) 

	if ( job_runs_successful ): 

		#file_ready = isobars_Output_Ready(predict_sub_folders,"dump.volume",num_configs=400) 
		pass

	else:

		print "LAMMPS Jobs fails; Check errors "

		sys.exit()
	
	natoms = 512

	scaled_isobar = Isobar_Matching(natoms,predict_address["isobar"],UNITS)  

	costfunclogger.debug("Finish Reading output from LAMMPS ... ") 	

	return scaled_isobar 

if ( __name__) == "__main__": 

	# ----------  Commmand Line argument -------------------

	logger,TOTAL_CORES,INPUTFILES,JOBID = ReadCommandLine.Finish()  

	# ----------  Parse Input file ---------------------

	in_para = ParseInputFile(INPUTFILES) 

	# ----------- Set up working folders ---------------------

	HOME,ref_address,predict_address = Setup(JOBID,in_para.matching).Finish() 	

	# ------------- Initialize Simulations -------------------

	LAMMPS = simulation.Invoke_LAMMPS(in_para,predict_address,TOTAL_CORES,HOME)	

	# ------------- Initialize Output --------------------------- 

	write = Output(in_para,JOBID) 

	# -------------- Initialize Matching -----------------------

	UNITS = in_para.UNITS

	natoms = 512 
	
	N_tol_all,T = Initialize_Isobar_Matching(in_para) 

	ref_density, ref_density_norms, T_rank_ref = Compute_Isobar_Norm( ref_address,natoms ) 

	w = 1 

	ComputeObjective_by_Lammps(w,force_field_parameters=np.array([1,1,0,83225.62784858546, 16.83150386560724, -0.4996224400188112, 0.7855353379110108, 1.0599904040697514e-06, 2.313765475886687, 11404.761095439186,3.4883864236198705, 0.2832924250068106, 2.872130883059371, 39135.28415110749])) 

