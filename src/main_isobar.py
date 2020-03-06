# Standard Library: 
import numpy as np 
import subprocess 
import multiprocessing as mp 
import logging
import itertools 
import time
import os
import sys 

# Custom library: 

import reader 
import computerdf
import const_mod 
import sys_mod 
import simulation_engine_mod as simulation

class IO():

	def WriteBestCostFunc(self,itera,bestcost): 
		
		with open("Bestcostfunc_vs_itera.txt","a") as costfunc: 

			costfunc.write (zip(itera,bestcost) ) 
					 
	# Write Restart  

	def WriteRestart(self,filename,itera,restartfreq,best_parameter,fit_and_fix,constraints,simplex): 

		keyword = "Restart"

		if ( itera % restartfreq == 0): 

			best_and_fix = np.zeros(fit_and_fix.size) 

			best_and_fix[fit_and_fix==1] = best_parameter ; best_and_fix[fit_and_fix==0] = guess[fit_and_fix==0]   

			with open(filename,"w") as f: 
			
				RestartContent(f,itera,restartfreq,keyword,best_and_fix,fit_and_fix,constraints,simplex)	

		return None 

	# Write Log Restart 

	def WriteRestartLog(self,filename,freq,itera,best_parameter,fit_and_fix,constraints,simplex): 

		keyword = "Restart" 	

		if ( itera ==0 ): 

			open(filename,"w").close() 

		if ( itera % freq == 0 and itera != 0): 

			best_and_fix = np.zeros(fit_and_fix.size) 

			best_and_fix[fit_and_fix==1] = best_parameter ; best_and_fix[fit_and_fix== 0] = guess[fit_and_fix==0] 

			with open(filename,"a") as f:  
					
				RestartContent(f,itera,restartfreq,keyword,best_and_fix,fit_and_fix,constraints,simplex) 	

		return None  

	pass	

class CoarsedGrained(): 

	def __init__(self,num_mol): 

		# 3D coordinate in Corased-grained system

		self.n_dof_cg = 3 

		# 3D coordinate in Reference system 

		self.n_dof_ref = 3 

		# subtract linear momentum  

		self.extra_dof_cg = 3 

		self.extra_dof_ref = 3 

		self.n_total_cg = self.n_dof_cg*num_mol - self.extra_dof_cg 

		self.n_total_ref = self.n_dof_ref*num_mol - self.extra_dof_ref 

	def KineticEnergyLoss(self,num_mol,temp_ref):  

		kb = const_mod.kb  #1.38064852*10**(-23)  

		kinetic_ref =  0.5*((self.n_dof_ref*(num_mol)-self.extra_dof_ref)*temp_ref*kb) # Joule 

		ke_loss  =  0.5*(kb*temp_ref)*( self.n_total_ref - self.n_total_cg )   

		return ke_loss,kinetic_ref 

class TrainingData(): 

	def __init__(self,fileaddresss,outputaddress,selectconfig): 

		pass  
	
class Costfunc(): 

	pass 

class NelderMeadSimplex:
	
	def __init__(self,vertices,obj,NVertices):

		self.vertices = vertices

		self.cost = obj
	
		self.nvertices = NVertices 

		self.nparameter = NVertices - 1  

	def TransformationCoeff(self,keyword): 	

		if ( keyword == "standard" ): 

			alpha = 1 ; kai =  2.0 ; gamma = 0.5 ; sigma = 0.5 

			return alpha , kai,  gamma , sigma
			
		elif ( keyword == "adaptive" ): 

			alpha = 1  ; kai =  1+ 2.0/self.nparameter  ; gamma = 0.75-1.0/(2*self.nparameter)  ; sigma = 1-1.0/self.nparameter 

		return alpha,kai, gamma, sigma 

	def RecevieCostfunc(self,costfunc):  

		self.cost = costfunc 

		self.SortVertices() 	
		
		return None 

	def SortVertices(self): 
		
		low_to_high = np.argsort(self.cost) 

		self.vertices = self.vertices[low_to_high,:]
		
		self.cost = self.cost[low_to_high]

		self.worstvertex = self.vertices[-1,:]

		self.secondworstvertex = self.vertices[-2,:] 

		self.bestvertex = self.vertices[0,:]

	def FindCentroid(self):  

		except_worst = self.vertices[:-1,]	

		self.centroid = np.sum(except_worst,axis=0)/( self.vertices[:,0].size-1) 
			
		return self.centroid 
		
	def ReflectWorstVertex(self): 

		self.reflected = self.centroid + alpha*(self.centroid - self.worstvertex) 

		return self.reflected  

	def ReflectedVertexExpand(self):  

		self.expanded = self.centroid + kai*(self.reflected - self.centroid ) 

		return self.expanded  

	def ReflectedVertexOutsideContracted(self):  

		self.outsidecontracted = self.centroid + gamma*(self.reflected - self.centroid)
		
		return self.outsidecontracted  
		
	def ReflectedVertexInsideContracted(self): 	

		self.insidecontracted = self.centroid + gamma*( self.worstvertex - self.centroid)
	
		return self.insidecontracted 
	
	def ShrinkAllVerticesExceptBest(self):   

		self.shrinkvertices = np.zeros(( nvertices-1,nvertices-1) )

		for i in xrange(self.vertices-1): 
	
			self.shrinkvertices[i,:] = self.bestvertex + sigma(self.vertices[i+1,:]- self.bestvertex) 	

		return self.shrinkvertices 

def WriteCostfunction(itera,*costfuncs): 

	filename = "matching_cost_functions_per_iteration"+args["job"] + ".txt"
	
	writecostlogger = logging.getLogger("In WriteCostfunction") 

	list_of_costf = np.array([ cost for cost in costfuncs ]) 

	if ( itera == 0 ): 

		writecostlogger.debug("Clear the cost functions output files when starting optimization ...")

		open(filename,"w").close()  

		writecostlogger.debug("Finish clearing  the cost functions output files when starting optimization ...")

	else: 

		with open(filename,"a") as output: 

			np.savetxt(output, list_of_costf,newline=" ",fmt="%.4e" ) 	

			output.write("\n") 
	
	return None 
			
def WriteCostFunc(itera,bestcost,worstcost): 

	if ( itera == 0 ): 

		open("bestcostfunc_vs_itera.txt","w").close() 

		open("worstcostfunc_vs_itera.txt","w").close() 

	else: 

		with open("bestcostfunc_vs_itera.txt","a") as costfunc: 

			results = np.array([itera,bestcost]) 

			np.savetxt(costfunc, results, newline = " " , fmt = "%.6e")  
			
			costfunc.write("\n") 

		with open("worstcostfunc_vs_itera.txt","a") as costfunc: 

			results = np.array([itera,worstcost]) 

			np.savetxt(costfunc, results, newline = " " , fmt = "%.6e")  
			
			costfunc.write("\n") 

	return None 

def RestartRead(parafile,objfile):
		
	paraspace = np.loadtxt(filename) 	

	vertices_obj = np.loadtxt(objfile) 
		
	return paraspace,vertices_obj  

def RestartContent(f,itera,restartfreq,keyword,best_parameter,fit_and_fix,constraints,simplex):  

	contents = "& Iteration: " + str(itera) + "\n"

	bestparameter = "& Guess initial Parameters: \n"

	f.write(contents) 

	f.write("\n")

	f.write(bestparameter) 

	f.write("\n")

	np.savetxt(f,best_parameter,newline=" ",fmt="%.10f") 

	f.write("\n") 
	f.write("\n") 

	f.write("& fit (1) and fix (0) parameters: ") 

	f.write("\n") 
	f.write("\n") 

	np.savetxt(f,fit_and_fix,newline=" ",fmt="%d") 
	
	f.write("\n") 
	f.write("\n") 

	f.write("& constraints: ")

	f.write("\n") 
	f.write("\n") 

	np.savetxt(f,constraints,newline=" ", fmt="%s") 		

	f.write("\n") 
	f.write("\n") 

	f.write("& create (Perturb) or use existing vertices (Restart): ") 
	f.write("\n") 

	f.write(keyword) 
	f.write("\n" ) 

	np.savetxt(f,simplex.cost,newline=" ",fmt="%.4e") 

	f.write("\n") 
	f.write("\n") 

	np.savetxt(f,simplex.vertices,fmt="%.10f") 

	f.write("\n") 
	f.write("\n") 

	return None  

def SelectRunMode(arg): 

	mode = { 
	
	"debug": logging.DEBUG, 
	"run": logging.INFO

	}  
	
	mode.get(arg,"invalid") 

	if ( mode[arg] == "invalid"):  

		print "Invalid Run mode: Choose debug or run"

		sys.exit()  

	return mode[arg] 

def SelectFormatter(arg): 

	mode = { 
	
	"debug": "%(asctime)s - %(name)s - %(levelname)s - %(message)s", 
	"run": "%(message)s" 
	}  

	mode.get(arg,"invalid") 
	
	if ( mode[arg] == "invalid"): 

		print "Invalid Run mode: Choose debug or run"

		sys.exit()  
	
	return mode[arg] 
	
def SetRunMode(logname,mode): 

	logger = logging.getLogger() 

	logger.setLevel(SelectRunMode(mode)) 

	fh = logging.FileHandler(logname,mode="w")

	formatter = logging.Formatter(SelectFormatter(mode)) 

	fh.setFormatter(formatter) 

	logger.addHandler(fh) 

	return logger

def WriteRestart(filename,itera,restartfreq,best_parameter,fit_and_fix,constraints,simplex): 

	keyword = "Restart"

	if ( itera % restartfreq == 0): 

		best_and_fix = np.zeros(fit_and_fix.size) 

		best_and_fix[fit_and_fix==1] = best_parameter ; best_and_fix[fit_and_fix==0] = guess[fit_and_fix==0]   

		with open(filename,"w") as f: 
		
			RestartContent(f,itera,restartfreq,keyword,best_and_fix,fit_and_fix,constraints,simplex)	

	return None 

def WriteRestartLog(filename,freq,itera,best_parameter,fit_and_fix,constraints,simplex): 

	keyword = "Restart" 	

	if ( itera ==0 ): 

		open(filename,"w").close() 

	if ( itera % freq == 0 and itera != 0): 

		best_and_fix = np.zeros(fit_and_fix.size) 

		best_and_fix[fit_and_fix==1] = best_parameter ; best_and_fix[fit_and_fix== 0] = guess[fit_and_fix==0] 

		with open(filename,"a") as f:  
				
			RestartContent(f,itera,restartfreq,keyword,best_and_fix,fit_and_fix,constraints,simplex) 	

	return None  

def InputSettings_Is_Correct(set_parameter,constrain_index,zero_index):  

	# mode: perturb or restart 

	mode =set_parameter[3][0]
	
	if ( mode != "Restart" and  mode != "Perturb"): 

		print "Select the mode: Restart or Purturb to Initialize NelderMead Simplex" 

		sys.exit() 

	if ( len(set_parameter[0]) != len(set_parameter[1]) ): 
		
		print "The number of parameters provided is not equal to fitted parameters (=1) + unfitted parameters (=0) "

		sys.exit() 

	if ( mode =="Perturb" and set_parameter[1].count("1") != len(set_parameter[4] ) ):    
		 
		print "Simplex Size ERROR: The number of fitted parameters (=1) should be equal to number of perturbed parameters "

		sys.exit()  

	if ( mode=="Restart" and set_parameter[1].count("1") != len(set_parameter[4] )-1 ): 

		print "Simplex Size ERROR: The number of fitted parameters + 1  (=1) should be equal to number of vertices "

		sys.exit()  

	if ( len(set_parameter[2])%3  !=0 ): 

		print "Constraints ERROR: index, lowerbound, upperbound should be provided together" 

		sys.exit() 

	if (np.amax(constrain_index) > len(set_parameter[0]) -1 ) : 

		print "Constraints ERROR: The constraints index provided is out of bound" 

		sys.exit() 

	# Check constraints index

	for cindex in constrain_index: 

		if ( cindex in zero_index): 

			print "Constrains ERROR: Constraint index has to be fitted variable (=1). Unfitted variable (=0) can not be constrained "
	
			sys.exit() 

	return None 

def ReadInputSettings(filename): 

	with open(filename,"r") as rawinput: 

		set_parameter = [ ] 

		for line in rawinput: 
	
			contents = line.split() 	

			if ( contents == [] ): 

				continue 

			elif ( "#" in contents[0] or "&" in contents[0] ):  

				continue 

			else: 

				set_parameter.append(contents)

	return set_parameter 

def ExtractInputSettings(set_parameter):  

	# save given parameters 

	parameters = np.array(set_parameter[0]).astype(np.float64) 

	# save fitted or unfitted parameters

	fit_and_unfit = np.array(set_parameter[1]).astype(np.int) 	

	# save the unfitted parameters 

	unfit = np.array([ i for i,x in enumerate(fit_and_unfit) if x ==0 ],dtype=np.int) 

	# save the fitted parameters

	fit = np.array([ i for i,x in enumerate(fit_and_unfit) if x == 1  ],dtype=np.int) 	

	# save the perturbation for creating each perturbation 

	keyword = set_parameter[3]

	stepsize = np.array(set_parameter[4]).astype(np.float64)

	# save the constraints index  

	constraints = np.array(set_parameter[2]) 

	num_constraints = len(constraints)/3 

	constraints_index = np.array([constraints[indx*3]  for indx in xrange(num_constraints)]).astype(np.int)-1  

	# Check input settings 

	InputSettings_Is_Correct(set_parameter,constraints_index,unfit) 

	for nindex in xrange(constraints_index.size): 

		num_shift = sum( i < constraints_index[nindex] for i in unfit)

		constraints_index[nindex] = constraints_index[nindex] - num_shift 	

	constraints_bound = np.array([ [ constraints[3*indx+1], constraints[3*indx+2]]   for indx in xrange(num_constraints)]) 
			
	return parameters,fit_and_unfit,keyword,constraints_index,constraints_bound

def PrintInputSettings(): 

	num_fit = guess[fit_and_unfit==1].size 
	
	num_fix = guess[fit_and_unfit==0].size

	unfit = np.array([ i for i,x in enumerate(fit_and_unfit) if x ==0 ],dtype=np.int)

	fit = np.array([ i for i,x in enumerate(fit_and_unfit) if x == 1  ],dtype=np.int)

	print "The number of Vertices: ", fit.size + 1  
	print "\n" 
	print "The input parameters are:  "
	print "\n" 
	print guess.tolist()  
	print "\n"
	print "The %s fitting parameters: " %(num_fit) 
	print "\n"
	print fit.tolist()   
	print "\n "
	print "The %s fixed parameters: "%(num_fix) 
	print "\n"
	print unfit.tolist()  
	print "\n"
	print "The %s constrained parameters: " %(constrains_index.size) 
	print "\n"	
	print (constrains_index).tolist() 
	
	return None  

def GetReferenceInformation(Reffile,skipfirst):  

	with open(Reffile) as coord: 

		filelines = sum(1 for x in coord) 
	
		coord.seek(0) 
		
		for line in coord: 	

			if ( line == "ITEM: NUMBER OF ATOMS\n"):	
				
				num_atoms = int((next(coord)))  

				for i in xrange(4): 

					coord.next() 	
		
				line_select = coord.next().split()  

				num_column = len(line_select[4:]) 	
				
				break 

		num_configs = filelines/( num_atoms + 9 )-skipfirst 	

	return num_atoms,num_column,num_configs 

def LammpsInitialization(potentialfiles,outputfiles):  

	# clear the contents of all output files 

	for output in outputfiles: 

		open(output,"w").close() 

	# clear the contents of lammps job 

	open( predict_folder + "jobfinish.txt", "w").close() 

	# check the potential files are there

	if ( not os.path.exists(potentialfiles)):   

		print "Lammps potential files are not found! "

		sys.exit() 
	
	# check if the Ref coord exists 

	if ( not os.path.exists(Reference_coord )): 

		print "Reference coord files not found! " 	
		
		sys.exit() 	

	if ( not os.path.exists(predict_folder + lammps_input)): 

		print "LAMMPS input files %s  not found" %lammps_input 
	
		sys.exit()  
	
	return None 

def CheckLammpsStatus(lammpslog):  

	with open(lammpslog,"r") as log: 

		line = log.readlines() 
		
		for words in line:  
	
			if ("ERROR" in words  ): 	
		
				print "Lammps got an error message: Check Lammps log file "
				
				sys.exit() 

			else: 
			
				print "No response from LAMMPS. Check if Lammps Finish or not"
	
				sys.exit() 
	
	return None 

def LammpsJobsFinish():

	counter = 0 

	while True:  
	
		if ( Empty( predict_folder + "jobfinish.txt") ): 

			time.sleep(0.1) 

			counter = counter + 1 

			if ( counter > 10000 ): 
			
				CheckLammpsStatus(lammpslog) 

			continue  

		else: 
		
			with open(predict_folder + "jobfinish.txt","r") as log:

				last = log.readlines()[-1].decode()

				#print last 
				if (  "JobDone!" in last): 

					break 

	return True 

def Empty(filename): 

	filesize = os.stat(filename).st_size 

	if ( filesize == 0):  

		emptyfile = True 

	else: 

		emptyfile = False

	return emptyfile 

def LammpsOutputReady(outputfiles):

	with open( predict_folder + "jobfinish.txt","a") as f:		

		while NotFinishDumping(outputfiles) :  
	
			time.sleep(0.02)

		print >> f, "Output files from LAMMPS are all ready "

	return True

def GatherLammpsOutput(num_atoms,num_configs,num_column,outputfiles): 

	engfiles = outputfiles[0] 

	forcefiles = outputfiles[1] 
		
	virialfiles = outputfiles[2]

	eng_dumped = np.loadtxt(engfiles,skiprows=1) 

	virial_dumped = np.loadtxt(virialfiles,skiprows=1) 	

	#force_dumped = ReadDumpedLammpsTraj(num_atoms,num_configs-1,num_column,forcefiles,skipfirst=0)  

	return eng_dumped,virial_dumped

def ReadDumpedLammpsTraj(num_atoms,num_configs,num_column,address,skipfirst): 

	with open(address) as coord: 
	
		allfiles = coord.readlines() 

		properties_list = np.empty(num_atoms*(num_configs-skipfirst)*num_column,dtype="float64") 

		counter = 0 

		for num in xrange(num_configs): 

			properties_begin  = 9*(num+1+skipfirst)+num_atoms*(num+skipfirst) 
		
			properties_finish = properties_begin + num_atoms  

			for line in allfiles[properties_begin:properties_finish]:  

				xyz = np.array(line.split()[2:]).astype(np.float64)  

				properties_list[counter:counter+num_column] = xyz 
									
				counter = counter + num_column

	return  properties_list  

def CountForceConfig(forcefiles): 

	count = 0 

	for line in open(forcefiles).xreadlines(): count += 1 

	num_dumped_config =  count/(num_atoms+9)

	return num_dumped_config 

def CountEenergyConfig(engfiles): 

	count = 0 

	eng = np.loadtxt(engfiles,skiprows =1)  

	return eng.size	

def CountVirialConfig(virialfile):

	count = 0 

	virial = np.loadtxt(virialfile,skiprows=1) 

	return virial.size

def NotFinishDumping(outputfiles): 
	
	notready = True 

	for output in outputfiles: 

		if ( output == predict_folder + "dump.eng"):  

			while Empty(output): 

				time.sleep(0.01) 
	
			dump_eng_config = CountEenergyConfig(output)
			
	
		elif ( output== predict_folder + "dump.force" ): 

			while Empty(output): 

				time.sleep(0.01) 

			dump_force_config = CountForceConfig(output)	
			
		
		elif ( output == predict_folder + "dump.virial" ): 
	
			while Empty(output): 

				time.sleep(0.01) 

			dump_virial_config = CountVirialConfig(output) 
			
	if ( dump_eng_config == num_configs and dump_force_config == num_configs and dump_virial_config == num_configs ): 

		notready  = False 

	return notready  

def not_finishing_rdf_traj(filename,ref_configs): 

	natoms,total_frames = reader.readdcdheader(dcdfile)  

	counter = 0 

	while ( total_frames != ref_configs ): 

		time.sleep(0.5) 

		natoms,total_frames = reader.readdcdheader(dcdfile)  

		counter = counter + 1 

		if ( counter >= 10000):  

			print """ERROR: RDF Trajectory does not finish dumping or number of  

			reference configurations used in gr calculations does not match  

			those predicted lammps ; check the dumping frequency and total run steps"""
				
			sys.exit() 

	return None 

def InitializeVertices(guess_vertices,cpercent,n_vertices): 
	
	parameters_dim = guess_vertices.size 
	
	vertices = np.zeros((n_vertices,parameters_dim)) 

	shift_vector = np.eye(parameters_dim) 

	vertices[0,:] = guess_vertices 

	for i in xrange(1,n_vertices):

		vertices[i,:] = guess_vertices + cpercent[i-1]*shift_vector[i-1,:]*guess_vertices[i-1] 

	return vertices

def ComputeRefEnergyForceAve_by_Jeremy_mW_Code(num_config,in_parameters):
	import computeforceinteraction 
	import readxyz   
	import readatom 

	forceall = np.zeros((force_ref.size)) ; energyAve = np.zeros(num_config)  

	for i in xrange(num_config):  
		
		filename=  "coord/Reference_mW_coord_"+str(i+1)+".xyz" 	

		n_atoms = readatom.read_atom(filename)  
		
		each_rows = n_atoms*3 

		xyz,box =  readxyz.read_xyz(filename,n_atoms) 
	
		force_1, eng_1 = computeforceinteraction.compute_interactions(in_parameters,xyz,box,n_atoms) 
	
		sindx = i*each_rows ; eindx = (i+1)*each_rows
		
		forceall[sindx:eindx] = np.reshape(force_1.T,(n_atoms*3))

		energyAve[i] = eng_1 

	return forceall,energyAve

#------------------------------------------------------------------------------------------- 
#        Radial Distribution Function Matching  
#--------------------------------------------------------------------------------------------

# Binning the cutoff distance; return the array of middle point of each interval  

def GetBins(num_bins,cutoff): 
	
	r_interval = cutoff/num_bins

	bins_position = np.zeros(num_bins) 
	
	for i in xrange(num_bins): 

		bins_position[i] = r_interval*0.5 + i*r_interval 
	
	return bins_position

# Write RDF histogram: 1st column is r ; 2nd column is gr  

def WriteRDF(filename,gr,radius):  

	np.savetxt(filename,zip(radius,gr))  

# Compute RDF histogram from a dcd trajectory by using serial job 

def ComputeRDF_bySerial(dcdfile,cutoff,num_bins): 

	natoms,total_frames = reader.readdcdheader(dcdfile) 

	total_atoms, total_volume = 0.0,0.0 

	radius = GetBins(num_bins,cutoff) 

	for current_frame in xrange(total_frames): 

		xyz,box = reader.read_dcd_xyzbox(dcdfile,natoms,current_frame) 
		
		vol = np.prod(box) 

		rdf_histogram =  computerdf.build_homo_pair_distance_histogram(natoms,cutoff,num_bins,xyz,box)

		total_volume = total_volume + vol 

		total_atoms = total_atoms + natoms 

	bulk_density = computerdf.bulk_density(total_atoms,total_volume) 

	gr = computerdf.normalize_histogram(rdf_histogram,num_bins,cutoff,natoms,bulk_density) 

	return gr  

# Determine the ranges of frames used by each core  

def SelectFrames(num_cores,total_frames): 

	frames_per_core = total_frames/num_cores 

	remainder = total_frames%num_cores

	ranges = []  

	for i in xrange(num_cores): 

		start = frames_per_core*i + 1    
            
		end = start + frames_per_core - 1  

		ranges.append([start,end])  

	ranges[-1][-1] = ranges[-1][-1] + remainder 
        
	return ranges 

def LanuchParallel_gr(num_cores,dcdfile,cutoff,num_bins): 

	p = mp.Pool(num_cores) 

	natoms,total_frames = reader.readdcdheader(dcdfile) 

	num_frames = total_frames % num_cores 

	ranges = SelectFrames(num_cores,total_frames) 
	
	results = [ p.apply_async( GetHistogram , args=(dcdfile,start,end,cutoff,num_bins,natoms )) for start,end in ranges ]

	gr = ComputeParallelgr(results,natoms,cutoff,num_bins) 

	p.close() 

	p.join() 
	
	return gr 

def ComputeParallelgr(results,natoms,cutoff,num_bins):

	total_volume = 0 ; total_atoms = 0 

	rdf_hist = np.zeros(num_bins) 

	for output in results: 	

		rdf_hist_each_core, atoms, volume =  output.get() 
		
		rdf_hist = rdf_hist + rdf_hist_each_core 

		total_volume = total_volume + volume 

		total_atoms = total_atoms + atoms

	num_configs = total_atoms/natoms	

	bulk_density = computerdf.bulk_density(total_atoms,total_volume) 

	gr = computerdf.normalize_histogram(rdf_hist,num_bins,cutoff,natoms,num_configs,bulk_density) 
	
	return gr  

def GetHistogram(dcdfile,start,end,cutoff,num_bins,natoms):  

	total_atoms, total_volume = 0.0,0.0 

	rdf_hist = np.zeros(num_bins ) 

	for current_frame in xrange(start,end+1): 

		xyz,box = reader.read_dcd_xyzbox(dcdfile,natoms,current_frame) 
		
		vol = np.prod(box) 

		rdf_hist = rdf_hist + computerdf.build_homo_pair_distance_histogram(natoms,cutoff,num_bins,xyz,box)

		total_volume = total_volume + vol 
		
		total_atoms = total_atoms + natoms 

	return rdf_hist, total_atoms, total_volume

# RDF Matching functional forms 


def RDFMatching(num_cores,dcdfile,cutoff,num_bins): 
	
	r_dist = GetBins(num_bins,cutoff) 

	gr_predict = LanuchParallel_gr(num_cores,dcdfile,cutoff,num_bins)

	r_interval = cutoff/num_bins 

	gr_ref = np.interp(r_dist,ref_gr[:,0],ref_gr[:,1])  

	WriteRDF("dump.gr",gr_predict,r_dist) 

	WriteRDF("Ref.gr",gr_ref,r_dist) 

	sqr_diff = ((r_dist*gr_predict - r_dist*gr_ref))**2  

	sum_rgr_diff  = 0.0 ; sum_ref =0.0 ; test =0.0 

	for index,y_diff in enumerate(sqr_diff) : 

		sum_rgr_diff  = sum_rgr_diff + r_interval*y_diff  
	
		sum_ref = sum_ref + r_interval*(r_dist[index]*gr_ref[index] -r_dist[index])**2  #(r_dist[index]*gr_ref[index] - r_dist[index])**2   

	scaled_gr = sum_rgr_diff/(sum_ref) 

	return scaled_gr 

#------------------------------------------------------------------------------------------- 
#        Isobars Matching  
#--------------------------------------------------------------------------------------------

def Compute_N_match(T_predict_sort,T_ref_sort):  

	N_match = 0 

	for N in zip(T_predict_sort,T_ref_sort): 

		if ( N[0] == N[1]): 

			N_match = N_match + 1 

	return N_match 

def Penalty( N_tol,N,T_predict_sort,T_ref_sort): 

	N_match = Compute_N_match(T_predict_sort,T_ref_sort) 

	penalty = max(N - N_tol - N_match, 0 )  

	return penalty 

def isobars_Output_Ready(sub_folders,filename,num_configs):

	counter =0 

	while True:

		done = 0

		for indx,address in enumerate(sub_folders):

			count = 0

			for line in open(address +"/"+ filename,"r"): count +=1
	
			# automatically skip first lines: count - 1	

			if ( num_configs != count-1 ):

				time.sleep(0.5)

				counter  = counter + 1

				if ( counter > 3000):  

					return False

			else:

				done += 1

		if ( done == len(sub_folders)):

			return True

def Get_Folders_Address(wk_folder): 

	home = os.getcwd() 

	os.chdir(wk_folder) 

	sub_folders = [ ] 

	sub_folders = Get_Subfolders(sub_folders) 

	num_jobs = len(sub_folders) 

	os.chdir(home) 

	return num_jobs,sub_folders 

def Get_Subfolders(sub_folders): 

	folders = next(os.walk('.'))[1] 

	if ( folders ): 

		 for folder in folders:

			os.chdir(folder) 

			Get_Subfolders(sub_folders) 

			os.chdir("../") 

	else: 

		sub_folders.append(os.getcwd()) 

	return sub_folders 

def Compute_Density(natoms,sub_folders,num_T,real_units):  	

	volume = np.zeros(num_T) 

	for indx,address in enumerate(sub_folders):

		volume[indx] = np.average( np.loadtxt(address+"/dump.volume" ) ) 	

	return  ( natoms/const_mod.NA)*const_mod.water_mol_mass/(volume*real_units.vol_scale*10**6)  

def Compute_Isobar_Matching_Reference(num_T,ref_sub_folders): 	

	ref_density = np.zeros(num_T) 

	for indx,address in enumerate(ref_sub_folders): 

		ref_density[indx] = np.average(np.loadtxt(address+"/Ref.density")[:,-1]) 
	
	return ref_density,np.var(ref_density) 
	
def Get_Reference_Predict_folders(ref_wk_folder,predict_wk_folder): 

	ref_num_jobs,ref_sub_folders = Get_Folders_Address(ref_wk_folder)

	predict_num_jobs,predict_sub_folders = Get_Folders_Address(predict_wk_folder)	

	return ref_num_jobs, predict_num_jobs, ref_sub_folders,predict_sub_folders     
	
def Isobar_Matching(natoms,sub_folders,num_T,real_units):

	predict_density = Compute_Density(natoms,sub_folders,num_T,real_units) 	

	indx = np.argsort(predict_density) 

	T_predict_sort = T[indx] 

	diff = predict_density - reference_density

	penalty = Penalty( N_tol,num_T,T_predict_sort,T_ref_sort) 

	scaled_isobar = np.sum((diff)**2/ref_density_norms) 

	print "penalty: ", penalty
			
	return scaled_isobar*penalty + scaled_isobar	
	
# Energy Matching

def ComputeScaledEnergy(eng_ref,eng_para): 

	ave_diff = np.average(eng_para - eng_ref )   

	diff = eng_para - eng_ref 

	relative_eng = ( diff -ave_diff )**2 

	eng_var = np.var(eng_ref) 

	scaled_eng = relative_eng/eng_var 

	return np.average(scaled_eng)  

def ComputeScaledForce(force_ref,force_para): 

	diff = force_para - force_ref

	cov_F_ref = np.var(force_ref) 

	scaled_force = np.matmul( diff.T*(1/cov_F_ref),diff)/(force_para.size)
	
	return scaled_force 

def ComputeScaledPressure(virial_ref,virial_predict): 

	P_virial_diff = ( virial_predict - virial_ref )  # units: Atm  

	P_ke_diff = ComputePressureChangeFromKE(num_atoms,temp_ref) 	

	Pressure_ave_diff = np.average( P_virial_diff + P_ke_diff)

	Pressure_diff = (P_virial_diff + P_ke_diff )

	relative_Pressure_diff = (Pressure_diff-Pressure_ave_diff)**2	

	norm_variances = np.var(pressure_ref) 

	scaled_virial = relative_Pressure_diff/norm_variances 

	return np.average(scaled_virial)

def ComputePressureChangeFromKE(num_mol,temp_ref): 

	kb = const_mod.kb  # kb = 1.38064852*10**(-23) J/K 

	ke_diff  = 0.5*(kb*temp_ref)*( n_total_dof_cg - n_total_dof_ref )

	return  ke_diff/vol_ref   

def ComputeSumSquared(force_ref_chunk,force_dump_chunk):  

	diff =  force_dump_chunk - force_ref_chunk  

	return np.sum(diff*diff)  
	
def ComputeVariances(sum_ref):  

	return sumq/(totalframes*num_atoms-1) 
	
def ComputeSmallChunkObjective(totalframes,chunksize,num_atoms,num_column,dumpfile,Reffile,skip_ref,skip_dump):  
	
	num_itera = totalframes/chunksize
			
	remainder = totalframes%chunksize

	sum_diff  = 0.0 

	sum_refforce = 0.0 ; sqr_ave = 0.0 

	if ( remainder != 0 ):

		print "Chunksize has to be divisible by total configurations"
		
		print "Chunksize is: ", chunksize, " and total configurations are: ", totalframes  

		sys.exit() 

	for each_chunk in xrange(num_itera): 	

		start_ref,end_ref = SelectReadingPositions(chunksize,each_chunk,num_atoms,skip_ref) 

		start_dump,end_dump = SelectReadingPositions(chunksize,each_chunk,num_atoms,skip_dump) 

		begin = time.time() 

		properties_ref, properties_dump = ReadSmallChunk(Reffile,dumpfile,chunksize,num_column,start_ref,start_dump,end_ref,end_dump) 	

		finish = time.time() 

		sum_diff = sum_diff + ComputeSumSquared(properties_ref,properties_dump)
	
		sum_refforce = sum_refforce + np.sum(properties_ref) 

		sqr_ave = sqr_ave + np.sum(properties_ref*properties_ref) 

	average_sqr = (sum_refforce/(totalframes*num_atoms*num_column))**2  
	
	sqr_average =  sqr_ave/(totalframes*num_atoms*num_column)  

	variances_ref = sqr_average - average_sqr 	
	
	return sum_diff/variances_ref/(totalframes*num_atoms*num_column)  

def ReadSmallChunk(Reffile,Dumpfile,chunksize,num_column,start_ref,start_dump,end_ref,end_dump): 

	properties_ref = np.zeros(chunksize*num_atoms*num_column,dtype=np.float64) 

	properties_dump = np.zeros(chunksize*num_atoms*num_column,dtype=np.float64)

	with open(Reffile, "r") as ref, open(Dumpfile,"r") as dump:  

		Refcontent = itertools.islice(ref,start_ref,end_ref) 

		Dumpcontent = itertools.islice(dump,start_dump,end_dump) 

		counter = 0 

		num_content = 2 + num_column 

		for Refline,Dumpline in itertools.izip(Refcontent,Dumpcontent):

			if ( len(Refline.split()) == num_content ) and (len(Dumpline.split()) == num_content): 

				force_ref = np.array(Refline.split()[2:]).astype(np.float32) 

				force_dump = np.array(Dumpline.split()[2:]).astype(np.float32) 

				properties_ref[counter:counter+num_column] = force_ref
	
				properties_dump[counter:counter+num_column] = force_dump

				counter = counter + num_column 
	
	return properties_ref, properties_dump 

def ComputeChunkObjective(total_configs,chunksize,num_atoms,num_column,dumpfile,Reffile,skip_ref,skip_dump):  
	
	num_itera = total_configs/chunksize
			
	remainder = total_configs%chunksize

	if ( remainder != 0 ):

		print "Chunksize has to be divisible by total configurations"
		
		print "Chunksize is: ", chunksize, " and total configurations are: ", total_configs  

		sys.exit() 

	lineskipped_ref = skip_ref*(num_atoms+9)

	lineskipped_dump = skip_dump*(num_atoms+9) 

	file_ref_ptr = GoToFilePosition(Reffile,lineskipped_ref) 

	file_dump_ptr = GoToFilePosition(dumpfile,lineskipped_dump)

	with open(Reffile,"r") as ref, open(dumpfile,"r") as dump:
	
		ref.seek(file_ref_ptr) ; dump.seek(file_dump_ptr) 

		gotonext = chunksize*( num_atoms +  9 )

		Ref_chunkdata = np.zeros(num_atoms*3*chunksize,dtype=np.float32)

		Dump_chunkdata = np.zeros(num_atoms*3*chunksize,dtype=np.float32)

		sum_refforce = 0.0 ; sqr_ave = 0.0 ; sum_diff  = 0.0  

		for i in xrange(num_itera):

			Ref_content = itertools.islice(ref,0,gotonext) ; Dump_content = itertools.islice(dump,0,gotonext) 

			counter = 0 

			for Ref_line,Dump_line in itertools.izip(Ref_content,Dump_content) :

				if ( len(Ref_line.split())== 5):

					Ref_chunkdata[counter:counter+3] = np.array(Ref_line.split()[2:],dtype=np.float32)
	
					Dump_chunkdata[counter:counter+3] = np.array(Dump_line.split()[2:],dtype=np.float32)

					counter = counter + 3
		
			sum_diff = sum_diff + ComputeSumSquared(Ref_chunkdata,Dump_chunkdata) 

			sum_refforce = sum_refforce + np.sum(Ref_chunkdata) 			

			sqr_ave = sqr_ave + np.sum(Ref_chunkdata*Ref_chunkdata) 

		average_sqr = (sum_refforce/(total_configs*num_atoms*num_column))**2 

		sqr_average =  sqr_ave/(total_configs*num_atoms*num_column) 

		variances_ref = sqr_average - average_sqr 
			
	return sum_diff/variances_ref/(total_configs*num_atoms*num_column) 

def ComputeVirial(total_configs,chunksize,num_atoms,num_column,dumpfile,Reffile,skip_ref,skip_dump):  
	
	num_itera = total_configs/chunksize
			
	remainder = total_configs%chunksize

	if ( remainder != 0 ):

		print "Chunksize has to be divisible by total configurations"
		
		print "Chunksize is: ", chunksize, " and total configurations are: ", total_configs  

		sys.exit() 

	lineskipped_ref = skip_ref*(num_atoms+9)

	lineskipped_dump = skip_dump*(num_atoms+9) 

	file_ref_ptr = GoToFilePosition(Reffile,lineskipped_ref) 

	file_dump_ptr = GoToFilePosition(dumpfile,lineskipped_dump)

	with open(Reffile,"r") as ref, open(dumpfile,"r") as dump:
	
		ref.seek(file_ref_ptr) ; dump.seek(file_dump_ptr) 

		gotonext = chunksize*( num_atoms +  9 )

		Ref_chunkdata = np.zeros(num_atoms*3*chunksize,dtype=np.float32)

		Dump_chunkdata = np.zeros(num_atoms*3*chunksize,dtype=np.float32)

		sum_dot = 0 	

		for i in xrange(num_itera):

			Ref_content = itertools.islice(ref,0,gotonext) ; Dump_content = itertools.islice(dump,0,gotonext) 

			counter = 0 

			for Ref_line,Dump_line in itertools.izip(Ref_content,Dump_content) :

				if ( len(Ref_line.split())== 5):

					Ref_chunkdata[counter:counter+3] = np.array(Ref_line.split()[2:],dtype=np.float32)
	
					Dump_chunkdata[counter:counter+3] = np.array(Dump_line.split()[2:],dtype=np.float32)

					counter = counter + 3
		
			sum_dot = sum_dot + np.dot(Ref_chunkdata,Dump_chunkdata)  

	return sum_dot  

def ReadFileIterator(fileaddress,start,end):  

    with open(fileaddress,"r") as itear: 

        content = itertools.islice(itear,start,end) 

        for each_line in content:  

            linedata = each_line.split() 

            if ( len(linedata)== 5):

                yield linedata[2:]

def ReadFileByChunk(fileaddress,start,end): 

    data_itera = ReadFileIterator(fileaddress,start,end) 

    return np.fromiter(itertools.chain.from_iterable(data_itera),dtype=np.float32)

def ComputeForceObjective(total_configs,chunksize,num_atoms,num_column,dumpfile,Reffile,skip_ref,skip_dump):  

	num_itera = total_configs/chunksize

	remainder = total_configs%chunksize

	if ( remainder != 0 ):

		print "Chunksize has to be divisible by total configurations"

		print "Chunksize is: ", chunksize, " and total configurations are: ", total_configs

		sys.exit()

	lineskipped_ref = skip_ref*(num_atoms+9)

	lineskipped_dump = skip_dump*(num_atoms+9)

	start_ref,start_dump = 0,0 ; end_ref,end_dump = 0,0  

	sum_refforce = 0.0 ; sqr_ave = 0.0 ; sum_diff  = 0.0  

	for i in xrange(num_itera):

		datasize = chunksize*( num_atoms+ 9 )

		start_ref = i*datasize + lineskipped_ref  

		end_ref = start_ref + datasize  

		start_dump = i*datasize + lineskipped_dump 

		end_dump = start_dump + datasize 

		Ref_chunkdata = ReadFileByChunk(Reffile,start_ref,end_ref)    
		
		Dump_chunkdata = ReadFileByChunk(dumpfile,start_ref,end_ref) 

		sum_diff = sum_diff + ComputeSumSquared(Ref_chunkdata,Dump_chunkdata) 

		sum_refforce = sum_refforce + np.sum(Ref_chunkdata) 

		sqr_ave = sqr_ave + np.sum(Ref_chunkdata*Ref_chunkdata) 

	average_sqr = (sum_refforce/(total_configs*num_atoms*num_column))**2

	sqr_average =  sqr_ave/(total_configs*num_atoms*num_column)

	variances_ref = sqr_average - average_sqr

	return sum_diff/variances_ref/(total_configs*num_atoms*num_column) 

def ComputeParallelForceObjective(total_configs,chunksize,num_atoms,num_column,dumpfile,Reffile,skip_ref,skip_dump):

    p = mp.Pool(2) 

    datafile = [ Reffile,dumpfile] 

    num_itera = total_configs/chunksize

    remainder = total_configs%chunksize

    if ( remainder != 0 ):

            print "Chunksize has to be divisible by total configurations"

            print "Chunksize is: ", chunksize, " and total configurations are: ", total_configs

            sys.exit()

    lineskipped_ref = skip_ref*(num_atoms+9)

    lineskipped_dump = skip_dump*(num_atoms+9)

    start_ref,start_dump = 0,0 ; end_ref,end_dump = 0,0

    sum_refforce = 0.0 ; sqr_ave = 0.0 ; sum_diff  = 0.0

    datasize = chunksize*( num_atoms+ 9 )

    for i in xrange(num_itera):

        start = i*datasize

        end = start + datasize

        results = [ p.apply_async( ReadFileByChunk, args=(data,start,end )) for data in datafile ]

        Ref_chunkdata,Dump_chunkdata = [array.get() for array in results]

        sum_diff = sum_diff + ComputeSumSquared(Ref_chunkdata,Dump_chunkdata)

        sum_refforce = sum_refforce + np.sum(Ref_chunkdata)

        sqr_ave = sqr_ave + np.sum(Ref_chunkdata*Ref_chunkdata)

    p.close()

    p.join()

    average_sqr = (sum_refforce/(total_configs*num_atoms*num_column))**2

    sqr_average =  sqr_ave/(total_configs*num_atoms*num_column)

    variances_ref = sqr_average - average_sqr

    return sum_diff/variances_ref/(total_configs*num_atoms*num_column)

def GoToFilePosition(fileaddress,lineskipped): 

	with open(fileaddress,"r") as skip:

		for i in xrange(lineskipped): 

			skip.readline()

		file_ptr = skip.tell() 

	return file_ptr 
	
def SelectReadingPositions(chunksize,each_chunk,num_atoms,skipfirst):  

	chunk_start = each_chunk*chunksize 

	skipcontent = skipfirst*(num_atoms +9) 
			
	start  = ( chunk_start*(num_atoms+9) + skipcontent)  	

	end = start + (num_atoms+9)*chunksize
		
	return start,end  

def ComputeObjective_by_Jeremy_mW_Code(w,num_config,in_parameters): 

	force_1,eng_1 = ComputeRefEnergyForceAve_by_Jeremy_mW_Code(num_config,in_parameters) 

	scaled_eng = ComputeScaledEnergy(eng_ref,eng_1) 

	scaled_force = ComputeScaledForce(force_ref,force_1) 

	return  w*scaled_eng + (1-w)*scaled_force 

def SubmitLammpsJobs(command,target_folder): 

	lammpslogger = logging.getLogger("In LAMMPS module") 

	lammpslogger.debug("Switch to target folders to Submit LAMMPS jobs ... ") 

	run_ck = os.getcwd() 

	os.chdir(target_folder) 

	predicted_ck = os.getcwd() 

	process = subprocess.Popen(command,stdout=subprocess.PIPE, shell=True)

	proc_stdout = process.communicate()[0].strip() 

	os.chdir(run_ck)

	lammpslogger.debug("LAMMPS jobs Finish and Swtich back to current working folder... ") 

	return None 

def ComputeObjective_by_Lammps(w,in_parameters,fixgroup):

	costfunclogger = logging.getLogger("In Cost function evalutations: ") 

	# Write the fitting potential parameters to separate files 
	
	WriteTersoffPotential(in_parameters,predict_sub_folders,fixgroup) 

	simulation.LAMMPS.Run()

	job_runs_successful = simulation.LAMMPS.Exit() 

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

	scaled_isobar = Isobar_Matching(natoms,predict_sub_folders,predict_num_jobs,real_units) 

	costfunclogger.debug("Finish Reading output from LAMMPS ... ") 	

	print scaled_isobar 

	return scaled_isobar 

def WriteCostfunctions(itera,list_of_costf): 
		

	return None 

def ComputeObjective_by_Lammps_Parallel(w,paraspace,fixgroup):  

	return 

def GenNelderMeadSimplex(fit_and_unfit, stepsize, guess):  
		
	guess_variable = guess[fit_and_unfit==1] ; fix_variable = guess[fit_and_unfit==0]  

	NVertices = guess_variable.size + 1  

	paraspace = InitializeVertices(guess_variable,stepsize,NVertices)
	
	funcvertices = ComputeFuncVertices(paraspace,fit_and_unfit)              

	vertices_sorted,ordered_para = OrderVertices(funcvertices,paraspace)
	
	return NVertices,vertices_sorted,ordered_para  

def InitializeNelderMeadSimplex(guess,fit_and_unfit):  
	
	keyword = set_parameter[3][0]

	if ( keyword == "Perturb" or keyword == "perturb" ): 

		stepsize = np.array(set_parameter[4]).astype(np.float64)

		NVertices,vertices_sorted,ordered_para = GenNelderMeadSimplex(fit_and_unfit, stepsize, guess) 

		return NVertices,vertices_sorted,ordered_para 

	elif ( keyword =="Restart" or keyword =="restart"): 

		vertices_sorted = np.array(set_parameter[4]).astype(np.float64)  

		ordered_para = np.array(set_parameter[5:]).astype(np.float64)

		#vertices_sorted,ordered_para = OrderVertices(vertices,ordered_para)  

		NVertices = vertices_sorted.size 

		return NVertices,vertices_sorted,ordered_para 	

	else: 

		print "input files mode not recognized: Please type restart or perturb"

		sys.exit() 

		
def SelectNelderMeadParameters(keyword,num_parameter):  

	if ( keyword == "standard" ): 

		alpha = 1 ; kai =  2.0 ; gamma = 0.5 ; sigma = 0.5 

		return alpha , kai,  gamma , sigma
	 	
	elif ( keyword == "adaptive" ): 

		alpha = 1  ; kai =  1+ 2.0/num_parameter  ; gamma = 0.75-1.0/(2*num_parameter)  ; sigma = 1-1.0/num_parameter 

		return alpha , kai,  gamma , sigma 

def ComputeFuncVertices(vertices,fit_and_unfit): 

	num_vertices = vertices[:,0].size 

	funcvertices = np.zeros(num_vertices)

	for i in xrange(num_vertices): 

		in_parameters = vertices[i,:] 
		
		ApplyNelderMeadConstraints(in_parameters,constrains_index,constraints_bound)   	
		
		#funcvertices[i]  = ComputeObjective_by_Jeremy(w,num_config,in_parameters)   
		
		funcvertices[i]  = ComputeObjective_by_Lammps(w,in_parameters,fit_and_unfit) 

	return funcvertices 			

def OrderVertices(func,parameters): 

	iidx=  np.argsort(func) 

	parameters  = parameters[iidx,:] 

	funcvertices = func[iidx] 

	func= funcvertices

	return func,parameters

def ComputeCentroid(parameters,worst_i): 

	except_worst = parameters[:-1,:] 

	middle = np.sum(except_worst,axis=0)/( parameters[:,0].size-1)  

	return middle

def NelderMeadCentroid(vertices): 

	except_worst = vertices[:-1]	
	
	middle = np.sum( except_worst )/except_worst.size 
		
	return middle  

def NelderMeadReflection(centroid,worst_x): 

	vr = centroid + alpha*(centroid - worst_x) 

	return vr 	

def NelderMeadExpansion(centroid,vr_x): 

	ve_x = centroid + kai*(vr_x-centroid)

	return ve_x

def NelderMeadContractionO(centroid,vr_x): 

	vc_out_x = centroid + gamma*(vr_x-centroid) 

	return vc_out_x 

def NelderMeadContractionI(centroid,vr_w_x): 	

	vc_in_x = centroid + gamma*( vr_w_x - centroid) 

	return vc_in_x

def NelderMeadShrinkVertices(lambda_x,nvertices): 

	x_shrink = np.zeros(( nvertices-1,nvertices-1) ) 

	for i in xrange(nvertices-1): 

		v_shrink = lambda_x[0,:] + sigma*( lambda_x[i+1,:] - lambda_x[0,:])   

		x_shrink[i,:] = v_shrink	

	return x_shrink 	

def ApplyNelderMeadConstraints(array,position,criterion): 

	num_v = position.size 

	num_criterion = np.size(criterion,0)  

	if ( num_v == num_criterion and num_v > 0 ):  

		for i in xrange(num_criterion):  

			lower = criterion[i][0] ; upper = criterion[i][1] 

			constraints_lower  = lower + "<=" + str(array[position[i]]) 

			constraints_upper  =  str(array[position[i]]) + "<=" + upper  
		
			if ( eval(constraints_lower)): 

				pass   

			else:  

				print "Lower constraints are applied..."
		
				array[position[i]] = lower 

			if ( eval(constraints_upper)): 

				pass   

			else:  
			
				print "Upper constraints are applied..." 	

				array[position[i]] = upper 

	return None    

def ApplyNelderMeadConstraints_old(vertices,position,criterion): 

	if ( use_constraints): 

		num_v = position.size 

		num_criterion = len(criterion)  
		
		if ( num_v == num_criterion):   
			
			for i in xrange(num_v):  
		
				constraints = str(vertices[position[i]]) + criterion[i] 	
			
				if ( eval(constraints)): 

					continue  

				else:  

					print "Constraints are applied........" 

					vertices[position[i]] =criterion[i][-1] 

		elif ( num_v > num_criterion and num_criterion == 1): 

			for i in xrange(num_v): 

				constraints = str(vertices[position[i]]) + criterion[0] 

				if ( eval(constraints)): 

					continue 

				else: 
			
					print "Constraints are applied........" 
			
					vertices[position] = criterion[0][-1] 

					return None    

		else: 
				
			print "something wrong with Constraints.........." 

			sys.exit() 

	else: 

		return None    

def PrintNelderMeadInitialization(): 
	print "\n" 
	print "The Nelder-Mead Parameters, alpha,kai,gamma,sigma are: "
	print "The Reflection parameter: ", "alpha =", alpha
	print "The Expansion parameter: ", "kai =", kai 
	print "The Contraction parameter: ", "gamma =", gamma 
	print "The Shrinkage parameter: ", "sigma =", sigma 
	print "\n" 
	
	print "\n" 
	print "Initial Vertices : "
	print "\n" 

	for i in xrange(vertices_sorted.size): 

		print "Vertex:  ", i+1  
		print "Parameters:  ", ordered_para[i,:].tolist()  
		print "Objective value: ", vertices_sorted[i] 
		print "\n"

	print "\n" 
	print "Search for minimum begins........" 
	print "\n" 
	print "\n" 

	return None

def PrintNelderMeadRuntime(ordered_para,vertices_sorted): 
	
	print "Best vertex: "
	print "Parameters: ", ordered_para[0,:].tolist() 
	print "Objective function: ", vertices_sorted[0] 
	print "\n"
	
	print "Worst vertex: "
	print "Parameters: ", ordered_para[-1,:].tolist() 	
	print "Objective function: ", vertices_sorted[-1] 
	print "\n"

	return None 
	
def WriteStillingerWebPotential(potential,filename,fixgroup): 	

	element = ["W","W","W"]

	atomtype = " ".join(element) + " "  

	fix_index_var = np.argwhere(fixgroup == 1 ) 	
	
	fix_index_fix = np.argwhere(fixgroup == 0  )

	potential_out = np.zeros(fixgroup.size)   

	potential_out[fix_index_fix] = guess[fix_index_fix] 

	potential_out[fix_index_var[:,0]] = potential 
		
	#potential_out = np.append(potential_out,"0.0") 

	pair = atomtype + " ".join(potential_out.astype(str))  

	with open(filename,"w") as output:

		output.write(pair) 

	return None  

def WriteTersoffPotential(potential,folders_address,fixgroup):	

	element = ["WT","WT","WT"]

	atomtype = " ".join(element) + " "  

	fix_index_var = np.argwhere(fixgroup == 1 ) 	
	
	fix_index_fix = np.argwhere(fixgroup == 0  )

	potential_out = np.zeros(fixgroup.size)   

	potential_out[fix_index_fix] = guess[fix_index_fix] 

	potential_out[fix_index_var[:,0]] = potential 
		
	#potential_out = np.append(potential_out,"0.0") 

	pair = atomtype + " ".join(potential_out.astype(str))  

	for folder in folders_address:

		filename = folder +"/" + "WT_ML-BOP.tersoff"

		with open(filename,"w") as output:

			output.write(pair) 

			time.sleep(0.02) 

	return None  

def ReinitializeSearch(bestvertices,NVertices,fixgroup):  

	bestvalues = ComputeObjective_by_Lammps(w,bestvertices,fixgroup) 

	save_object = np.zeros(bestvertices.size*2)  

	save_vertices = np.zeros((bestvertices.size*2,bestvertices.size))  
 
	counter = 0 

	for sign in [-1,1]: 

		for ele in xrange(bestvertices.size):  

			perturb = np.copy(bestvertices)

			perturb[ele] = perturb[ele] + perturb[ele]*0.1*sign   

			save_object[counter] = ComputeObjective_by_Lammps(w,perturb,fixgroup ) 			

			save_vertices[counter,:] = perturb  
	
			counter = counter + 1 

	#save_object = np.insert(save_object,-1,bestvalues) 

	#save_vertices = np.insert(save_vertices,-1,bestvertices, axis=0) 
	
	indexx = np.argsort(save_object) 

	best_n_plus_1_index = indexx[0:NVertices]

	reinitial_vt  = save_vertices[best_n_plus_1_index,:] 

	reinitial_obj = save_object[best_n_plus_1_index] 

	return reinitial_obj,reinitial_vt

def AdjustVerticesSize():

	return None  

def NelderMeadTermination(N_itera): 

	# variances are small
	
	# some vertices are equal

	return None  

def IterationCounter(input_para): 

	if ( input_para.keyword == "Restart"): 
					
		return input_para.counter 				

	elif ( input_para.keyword =="Perturb"): 

		return 0  	

#------------------------------------------------------------------------------------------------
#---------------------------------      Main     ------------------------------------------------
#------------------------------------------------------------------------------------------------ 

#-------- Read Optimization setup-----------------

args = sys_mod.take_command_line_args() 

real_units = const_mod.Units("Real")

w = 1 

#-------------------- Isobar parameters -----------------------------

N_tol = 1 

matching_type = "isobar" 

predict_folder = "../Predicted_jobID_" + args["job"]+ "/"

ref_wk_folder = "../ReferenceData/" + matching_type + "/"

working_folder = predict_folder + matching_type + "/" 

os.mkdir(predict_folder)

os.mkdir(working_folder)

cores = args["cores"] 

simulation.LAMMPS(matching_type,working_folder,"module load intel/psxe-2019-64-bit && srun -n %d --mpi=pmi2 lmp_ml_water < in.isobar",cores)

ref_num_jobs,predict_num_jobs,ref_sub_folders,predict_sub_folders = Get_Reference_Predict_folders( ref_wk_folder,working_folder ) 

reference_density, ref_density_norms = Compute_Isobar_Matching_Reference(ref_num_jobs,ref_sub_folders) 

T = np.array([260,270,280,290,300,310])

indx = np.argsort(reference_density) 

T_ref_sort = T[indx] 

#--------------------------- Optimization System  ----------------------------

log = SetRunMode("run_optimization_" + args["job"]+".log","debug") 

set_parameter = ReadInputSettings("in.optimize")    

guess,fit_and_unfit,keyword,constrains_index,constraints_bound = ExtractInputSettings(set_parameter) 

logrestart = "../Restart/log.restart"

currentrestart = "../Restart/current.restart"

restartfreq = 5  

PrintInputSettings() 

log.info("Start intializing the Nelder Mead Simplex ..." ) 

NVertices,vertices_sorted,ordered_para = InitializeNelderMeadSimplex(guess,fit_and_unfit ) 

log.info("Finish intializing the Nelder Mead Simplex ...") 

simplex = NelderMeadSimplex(ordered_para,vertices_sorted,NVertices) 

alpha,kai,gamma,sigma = SelectNelderMeadParameters("adaptive",NVertices-1)

PrintNelderMeadInitialization() 

#--------------------- Minimization Iteration Begins Here------------------------- 
log.info("Nelder-Mead Optimization begins ...") 

for itera in xrange(20000):

	log.info("\n") 
	log.info("-------------------------------------------Current iteration-------------------------------------------\n") 
	log.info("                                                 %d                                                    " %itera) 
	log.info("-------------------------------------------------------------------------------------------------------\n")  

	#WriteCostfunction(itera,each_scaled_eng,each_scaled_force,each_scaled_virial) 
	#WriteCostfunction(itera,each_scaled_gr) 

	#WriteCostFunc(itera,simplex.cost[0],simplex.cost[-1]) 
	
	worst = vertices_sorted.size -1  
	
	second_worst = vertices_sorted.size-2  

	best = 0  

	except_best = best + 1  

	# compute centroid	

	log.info("Compute the centroid of all vertices except the worst vertex ...\n") 

	centroid = ComputeCentroid(ordered_para,worst)

	# Reflection Operation: Relect the worst vertices

	log.info("Reflect the worst (with the highest objective function) vertex accross the centroid ... \n") 
	
	vr_x = NelderMeadReflection(centroid,ordered_para[worst,:])
	
	ApplyNelderMeadConstraints(vr_x,constrains_index,constraints_bound)

	# get objective function values at reflected vertices 

	fvr_x = ComputeObjective_by_Lammps(w,vr_x,fit_and_unfit)   

	if ( fvr_x < vertices_sorted[second_worst] and vertices_sorted[best] <= fvr_x):  
		
		ordered_para[worst,:] = vr_x 

		vertices_sorted[worst] = fvr_x  	

		log.info("Since the objective function of reflected vertex  is worst (higher) than best (lowest) vertex but better than that of second worst (second highest) vertex ... \n ") 
		log.info("The reflected vertex is accpeted and replace the worst vertex ...\n" )
		log.info( "Now sort all vertices values... \n" )  
		
		vertices_sorted,ordered_para = OrderVertices(vertices_sorted,ordered_para)

		PrintNelderMeadRuntime(ordered_para,vertices_sorted) 

		log.info("Start new iteration ... \n") 

		simplex.vertices =ordered_para ; simplex.cost = vertices_sorted 
		
		WriteRestart(currentrestart,itera,restartfreq,ordered_para[best,:],fit_and_unfit,set_parameter[2],simplex) 

		WriteRestartLog(logrestart,restartfreq,itera,ordered_para[best,:],fit_and_unfit,set_parameter[2],simplex)  

		continue 		

	# Expansion 
	
	if ( fvr_x < vertices_sorted[best]): 
		
		log.info( "The objective function value of reflected vertex is better (lower) than the best vertex ... \n" ) 
		log.info( "Then, Further explore the reflected direction by expansion from the current reflected vertex ... \n ") 

		ve_x = NelderMeadExpansion(centroid,vr_x) 

		ApplyNelderMeadConstraints(ve_x,constrains_index,constraints_bound)   	

		feve_x = ComputeObjective_by_Lammps(w,ve_x,fit_and_unfit) 

		if (  feve_x < fvr_x ): 
		
			log.info( "The objective function value of expanded vertex is better (lower) than value of reflected vertex ... \n" )  
	
			ordered_para[worst,:] = ve_x  

			vertices_sorted[worst] = feve_x  

			log.info( "The expanded vertex is accpeted and replace the worst vertex ... \n" )  
		
			log.info( "Now sort all vertices values ... \n")  
		
			vertices_sorted,ordered_para = OrderVertices(vertices_sorted,ordered_para) 

			PrintNelderMeadRuntime(ordered_para,vertices_sorted)

			log.info( "Start new iteration ... \n")  

			simplex.vertices =ordered_para ; simplex.cost = vertices_sorted 
			
			WriteRestart(currentrestart,itera,restartfreq,ordered_para[best,:],fit_and_unfit,set_parameter[2],simplex) 

			WriteRestartLog(logrestart,restartfreq,itera,ordered_para[best,:],fit_and_unfit,set_parameter[2],simplex)  

			continue 

		# Relection 

		else: 

			log.info( "The objective function value of expanded vertex is worst (higher) than value of reflected vertex ... \n" ) 

			ordered_para[worst,:] = vr_x 

			vertices_sorted[worst] = fvr_x 

			log.info( "Then, the new expanded vertex is not accepted and only reflected vertex is used to replace the worst vertex ... \n")   

			log.info( "Now sort all vertices values ... \n")  

			vertices_sorted,ordered_para = OrderVertices(vertices_sorted,ordered_para) 
		
			PrintNelderMeadRuntime(ordered_para,vertices_sorted) 
				
			log.info( "Start new iteration ... \n") 

			simplex.vertices =ordered_para ; simplex.cost = vertices_sorted 
			
			WriteRestart(currentrestart,itera,restartfreq,ordered_para[best,:],fit_and_unfit,set_parameter[2],simplex) 

			WriteRestartLog(logrestart,restartfreq,itera,ordered_para[best,:],fit_and_unfit,set_parameter[2],simplex)  

			continue 	

	# Contraction: 
	
	if ( fvr_x  >= vertices_sorted[second_worst] ): 

		log.info( "Reflected vertex (objective function) is worst (higher) than that of second-worst vertex ... \n") 

		if ( fvr_x < vertices_sorted[worst] ): 
			
			log.info( "Reflected vertex is better than that of worst vertex ... \n") 
			
			log.info( "So, keep the reflected vertex first and perform the outside contraction to shrink the reflected value toward the centroid ... \n" ) 
			
			# outside contraction	
			
			v_oc = NelderMeadContractionO(centroid,vr_x)
			
			ApplyNelderMeadConstraints(v_oc,constrains_index,constraints_bound)   	

			fc_outside = ComputeObjective_by_Lammps(w,v_oc,fit_and_unfit) 
			
			if ( fc_outside <= fvr_x ):

				log.info( "After the outside contraction, the resulting objective function of new vertex is better (lower) than that of reflected vertex ... \n" ) 
				
				log.info( "Then, the new contracted vertex is accpeted and replace the worst vertex ... \n") 
				
				log.info( "Now sort all vertices values ... \n")  

				ordered_para[worst,:] = v_oc 

				vertices_sorted[worst] = fc_outside 

				vertices_sorted,ordered_para = OrderVertices(vertices_sorted,ordered_para) 

				PrintNelderMeadRuntime(ordered_para,vertices_sorted) 

				log.info( "Start new iteration ... \n")  

				simplex.vertices =ordered_para ; simplex.cost = vertices_sorted 
				
				WriteRestart(currentrestart,itera,restartfreq,ordered_para[best,:],fit_and_unfit,set_parameter[2],simplex) 

				WriteRestartLog(logrestart,restartfreq,itera,ordered_para[best,:],fit_and_unfit,set_parameter[2],simplex)  

				continue 

			else:  

				log.info( "After the outside contraction, the resulting objective function of new contracted vertex is worse (higher) than that of reflected vertex ... \n" )  
				
				log.info( "Then, the shrinkage will are performed ... \n")   

				x_shrink = NelderMeadShrinkVertices(ordered_para,NVertices)
				
				vertices_shrinked = ComputeFuncVertices(x_shrink,fit_and_unfit)  

				vertices_sorted[1:] = vertices_shrinked		

				ordered_para[1:,:] = x_shrink 

				vertices_sorted,ordered_para = OrderVertices(vertices_sorted,ordered_para) 

				log.info( "Now sort all vertices values ... \n")  

				PrintNelderMeadRuntime(ordered_para,vertices_sorted) 

				log.info( "Start new iteration ... \n")   

				simplex.vertices =ordered_para ; simplex.cost = vertices_sorted 
				
				WriteRestart(currentrestart,itera,restartfreq,ordered_para[best,:],fit_and_unfit,set_parameter[2],simplex) 

				WriteRestartLog(logrestart,restartfreq,itera,ordered_para[best,:],fit_and_unfit,set_parameter[2],simplex)  

				continue 

		elif ( fvr_x >= vertices_sorted[worst]): 	
			
			# inside contraction 

			log.info( "Reflected vertex is even worse (higher) than that of the worst vertex ...  \n")  
			
			log.info( "Then, reject the reflected vertex and perform the inside contraction to bring the worst vertex toward the centroid ... \n" )   
		
			v_ic = NelderMeadContractionI(centroid,ordered_para[worst,:])	

			ApplyNelderMeadConstraints(v_ic,constrains_index,constraints_bound)   	
				
			fc_inside = ComputeObjective_by_Lammps(w,v_ic,fit_and_unfit)  

			if ( fc_inside < vertices_sorted[worst] ): 

				log.info( "Now, the objective function of new vertex is better than that of the worst vertex ... \n") 
				
				log.info( "Then, accept the new contracted vertex and replace the worst vertex with it ... \n")  

				ordered_para[worst,:] = v_ic

				vertices_sorted[worst] = fc_inside

				vertices_sorted,ordered_para = OrderVertices(vertices_sorted,ordered_para) 

				log.info( "Now sort all vertices values ... \n")  

				PrintNelderMeadRuntime(ordered_para,vertices_sorted) 

				log.info( "Start new iteration ... \n")  

				simplex.vertices =ordered_para ; simplex.cost = vertices_sorted 
				
				WriteRestart(currentrestart,itera,restartfreq,ordered_para[best,:],fit_and_unfit,set_parameter[2],simplex) 

				WriteRestartLog(logrestart,restartfreq,itera,ordered_para[best,:],fit_and_unfit,set_parameter[2],simplex)  

				continue

			else:	
				
				log.info( "Now, the objective function of new vertex is still worst than than that of the worst vertex ... \n") 
				
				log.info( "Then, the shrinkage are performed ... \n")   

				x_shrink = NelderMeadShrinkVertices(ordered_para,NVertices)  
	
				vertices_shrinked = ComputeFuncVertices(x_shrink,fit_and_unfit)

				vertices_sorted[except_best:] = vertices_shrinked

				ordered_para[except_best:,:] = x_shrink 

				vertices_sorted,ordered_para = OrderVertices(vertices_sorted,ordered_para) 
				
				log.info( "After shrinkage, now sort all vertices values ... \n")  
				
				PrintNelderMeadRuntime(ordered_para,vertices_sorted) 

				log.info( "Start new iteration ... \n")  

				simplex.vertices =ordered_para ; simplex.cost = vertices_sorted 
				
				WriteRestart(currentrestart,itera,restartfreq,ordered_para[best,:],fit_and_unfit,set_parameter[2],simplex) 

				WriteRestartLog(logrestart,restartfreq,itera,ordered_para[best,:],fit_and_unfit,set_parameter[2],simplex)  

				continue 

