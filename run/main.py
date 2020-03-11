# Standard Python Library 
import sys 
import os 
import logging
import time 
import subprocess 
import argparse
import multiprocessing as mp 
import itertools 
import random

# installed from conda:
import numpy as np  

# Custom Library 
sys.path.insert(0,"../src") 
import sys_mod 
import const_mod 
import simulation_engine_mod as simulation 
import reader
import computerdf 

def InitializeVertices(guess_vertices,cpercent,n_vertices): 
    
    parameters_dim = guess_vertices.size 
    
    vertices = np.zeros((n_vertices,parameters_dim)) 

    shift_vector = np.eye(parameters_dim) 

    # initial guess is autoamtically set as first vertice 

    vertices[0,:] = guess_vertices 

    for i in range(1,n_vertices):

        if ( guess_vertices[i-1] == 0): 
        
            new_vertices = guess_vertices + cpercent[i-1]*shift_vector[i-1,:]*0.05

        else: 

            new_vertices = guess_vertices + cpercent[i-1]*shift_vector[i-1,:]*guess_vertices[i-1] 

        ApplyNelderMeadConstraints(new_vertices,in_para.constraints_fit_index,in_para.constraints_bound)        

        vertices[i,:] = new_vertices 

    return vertices

def Initialize_Vertices_Spendley(guess_vertices,cpercent,n_vertices):

    parameters_dim = guess_vertices.size
    
    p = 1.0/(parameters_dim*np.sqrt(2))*((parameters_dim-1) + np.sqrt(parameters_dim+1))

    q = 1.0/(parameters_dim*np.sqrt(2))*( np.sqrt(parameters_dim+1) - 1)    

    design_matrix = np.zeros((parameters_dim,parameters_dim)) 

    vertices = np.zeros((n_vertices,parameters_dim))

    vertices[0,:] = guess_vertices

    for i in range(parameters_dim): 

        for j in range(parameters_dim):

            if ( i == j ): 

                design_matrix[i,j]   = p 

            else: 

                design_matrix[i,j]  = q 

    for i in range(1,n_vertices): 

        if ( guess_vertices[i-1] == 0): 

            new_vertices = guess_vertices + design_matrix[i-1,:]*0.05*cpercent[i-1]

        else: 

            new_vertices = guess_vertices + design_matrix[i-1,:]*guess_vertices*cpercent[i-1]

        ApplyNelderMeadConstraints(new_vertices,in_para.constraints_fit_index,in_para.constraints_bound)    

        vertices[i,:] = new_vertices 

    return vertices 

def GenNelderMeadSimplex(W,fit_and_unfit,stepsize,guess,constrains_index,constraints_bound):  
        
    guess_variable = guess[fit_and_unfit==1] ; fix_variable = guess[fit_and_unfit==0]  

    NVertices = guess_variable.size + 1  

    #paraspace = InitializeVertices(guess_variable,stepsize,NVertices)
 
    paraspace = Initialize_Vertices_Spendley(guess_variable,stepsize,NVertices)

    funcvertices = ComputeFuncVertices(W,paraspace) 

    vertices_sorted,ordered_para = OrderVertices(funcvertices,paraspace)
    
    return NVertices,vertices_sorted,ordered_para  

def InitializeNelderMeadSimplex(W,in_para): 
    
    if ( in_para.mode == "Perturb"):  

        fit_and_fix = in_para.fit_and_fix  

        guess = in_para.guess_parameter 
    
        constraints_bound = in_para.constraints_bound 

        constrains_index = in_para.constraints_index 

        stepsize = in_para.perturb

        NVertices,vertices_sorted,ordered_para = GenNelderMeadSimplex(W,fit_and_fix,stepsize,guess,constrains_index,constraints_bound)      

        return NVertices,vertices_sorted,ordered_para 

    elif ( in_para.mode =="Restart"):   

        vertices_sorted = np.array(in_para.obj).astype(np.float64)  

        ordered_para = np.array(in_para.vertices).astype(np.float64)    

        vertices_sorted,ordered_para = OrderVertices(vertices_sorted,ordered_para)  

        NVertices = vertices_sorted.size 

        return NVertices,vertices_sorted,ordered_para   

    else: 

        print ("Nelder-Mead Simplex mode not recognized: Please choose 'Restart' or 'Perturb' in the input file")

        sys.exit() 
        
def SelectNelderMeadParameters(keyword,num_parameter):  

    if ( keyword == "standard" ): 

        alpha = 1 ; kai =  2.0 ; gamma = 0.5 ; sigma = 0.5 

        return alpha , kai,  gamma , sigma
        
    elif ( keyword == "adaptive" ): 

        alpha = 1  ; kai =  1+ 2.0/num_parameter  ; gamma = 0.75-1.0/(2*num_parameter)  ; sigma = 1-1.0/num_parameter 

        return alpha , kai,  gamma , sigma 

def regroup_parameters(all_guess,fix_and_fit,para_fit): 

    para_all = np.arange(all_guess.size,dtype=np.float64) 

    para_fixed = all_guess[fix_and_fit==0] 

    para_all[fix_and_fit==0] = para_fixed  

    para_all[fix_and_fit==1]= para_fit 

    return para_all 

def ComputeFuncVertices(W,vertices): 

    num_vertices = vertices[:,0].size 

    funcvertices = np.zeros(num_vertices)

    for i in range(num_vertices): 

        in_parameters = vertices[i,:] 

        funcvertices[i] = ComputeObjective_by_Lammps(W,in_parameters) 

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

    vr_x = centroid + alpha*(centroid - worst_x) 
    
    ApplyNelderMeadConstraints(vr_x,in_para.constraints_fit_index,in_para.constraints_bound) 

    return vr_x 

def NelderMeadExpansion(centroid,vr_x): 

    ve_x = centroid + kai*(vr_x-centroid)

    ApplyNelderMeadConstraints(ve_x,in_para.constraints_fit_index,in_para.constraints_bound)

    return ve_x

def NelderMeadContractionO(centroid,vr_x): 

    vc_out_x = centroid + gamma*(vr_x-centroid) 

    ApplyNelderMeadConstraints(vc_out_x,in_para.constraints_fit_index,in_para.constraints_bound) 

    return vc_out_x 

def NelderMeadContractionI(centroid,vr_w_x):    

    vc_in_x = centroid + gamma*( vr_w_x - centroid) 

    ApplyNelderMeadConstraints(vc_in_x,in_para.constraints_fit_index,in_para.constraints_bound)

    return vc_in_x

def NelderMeadShrinkVertices(lambda_x,nvertices): 

    x_shrink = np.zeros(( nvertices-1,nvertices-1) ) 

    for i in range(nvertices-1): 

        v_shrink = lambda_x[0,:] + sigma*( lambda_x[i+1,:] - lambda_x[0,:])   

        ApplyNelderMeadConstraints(v_shrink,in_para.constraints_fit_index,in_para.constraints_bound)    

        x_shrink[i,:] = v_shrink    

    return x_shrink     

def ApplyNelderMeadConstraints(array,position,criterion): 

    logger = logging.getLogger("In Constraints Function: ") 

    num_v = position.size 

    num_criterion = np.size(criterion,0)  

    if ( num_v == num_criterion and num_v > 0 ):  

        for i in range(num_criterion):  

            lower = criterion[i][0] ; upper = criterion[i][1] 

            constraints_lower  = lower + "<=" + str(array[position[i]]) 

            constraints_upper  =  str(array[position[i]]) + "<=" + upper  
        
            if ( eval(constraints_lower)): 

                pass   

            else:  

                logger.info( "Lower constraints are applied...") 
                logger.info( "Parameter: " + str( array[position[i]]) + "  is constrained to " + str( lower))   
                
                array[position[i]] = lower 
                
            if ( eval(constraints_upper)): 

                pass   

            else:  
            
                logger.info( "Upper constraints are applied..." )   
                
                logger.info( "Parameter: " + str( array[position[i]]) + "  is constrained to " + str( upper))   
                
                array[position[i]] = upper 

    return None    

def PrintInputSettings(guess,fit_and_unfit,constrains_index,bounds,max_itera,obj_tol,para_tol): 

    num_fit = guess[fit_and_unfit==1].size 
    
    num_fix = guess[fit_and_unfit==0].size

    unfit = np.array([ i for i,x in enumerate(fit_and_unfit) if x ==0 ],dtype=np.int)

    fit = np.array([ i for i,x in enumerate(fit_and_unfit) if x == 1  ],dtype=np.int)

    log.info("\n \n") 
    log.info( "------------------------- Initialize Optimization Input Parameters -----------------------------------\n") 
    log.info( "Number of Vertices: %d \n"%(fit.size + 1 ))  
    log.info("-------------------------------------------------------------------------------------------------------\n")  
    log.info("Guess parameters are: \n " )  
    log.info(" ".join(str(para) for para in guess) + "\n" )  
    log.info("-------------------------------------------------------------------------------------------------------\n")  
    log.info("%d Fitting parameters are:  \n"  %(num_fit)) 
    log.info(" ".join(str(para) for para in guess[fit] ) + "\n")
    log.info("-------------------------------------------------------------------------------------------------------\n")
    log.info("%d fixed parameters are: \n"%(num_fix) ) 
    log.info(" ".join(str(para) for para in guess[unfit]) + "\n" )
    log.info("-------------------------------------------------------------------------------------------------------\n")
    log.info("Maximum iteration: %d \n"%max_itera)
    scif = "{0:.1e}".format(obj_tol) 
    log.info("objective function convergence tolerance: %s \n"%scif)
    scif = "{0:.1e}".format(para_tol)   
    log.info("force-field parameters convergence tolerance: %s \n"%scif)
    log.info("-------------------------------------------------------------------------------------------------------\n")  
    log.info("%d constrained parameters : \n" %(constrains_index.size)) 

    constraints_bounds = bounds.astype(np.float32) 

    for i in range(constrains_index.size): 

        log.info("The guess parameter: %.6f is constrained between %.6f and %.6f "%( float(guess[constrains_index[i]]),float(bounds[i][0]),float(bounds[i][1]) ) + "\n" )

    log.info("-------------------------------------------------------------------------------------------------------\n")
    if ( in_para.mode =="Perturb"): 
        log.info("%d step size is : \n"%in_para.perturb.size)  
        log.info(" ".join(str(step) for step in in_para.perturb) ) 
    log.info("-------------------------------------------------------------------------------------------------------\n") 
    log.info("\n \n") 

    return None  

#-----------------------------------------------------------------------------
#-------------------------- System Function --------------------------------
#-----------------------------------------------------------------------------

def output_data_is_ready(fileaddress,datatype,datafile,skip): 
    
    counter = 0 
    
    try: 

        nconfigs = int(open(fileaddress + "/" + "finish","r").readline())  
        
        while True: 
            
            num_configs_now = sys_mod.get_lines(fileaddress+ "/" + datafile)    

            if ((num_configs_now - skip)  == nconfigs): 

                break 

            else: 

                time.sleep(5) 
                
                counter = counter + 1  
        
                if ( counter > 400):  

                    print ("30 mins after simulation was invoked, output data is transfering ... " )
                    print ("Or the nconfigs output is not same as the read by ") 
    
                    sys.exit() 

    except IOError: 

        # That's fine if no finish file, just wait 1 seconds 
        
        time.sleep(1)   

    return None  

#-----------------------------------------------------------------------------
#-------------------------- Isobar Matching ----------------------------------
#-----------------------------------------------------------------------------

def Isobar_data_Ready(isobar_address): 

    for subfolder in isobar_address:

        for fileaddress in isobar_address[subfolder]: 

            output_data_is_ready(fileaddress,"isobar","dump.volume",skip=1) 

    return None 

def Compute_N_match(T_predict_sort,T_ref_sort):  

    N_match = 0 

    for N in zip(T_predict_sort,T_ref_sort): 

        if ( N[0] == N[1]): 

            N_match = N_match + 1 

    return N_match 

def Penalty( N_tol,T_predict_sort,T_ref_sort): 

    N = T_ref_sort.size 

    N_match = Compute_N_match(T_predict_sort,T_ref_sort) 

    penalty = max(N - N_tol - N_match, 0 )  

    return penalty 

def Parse_Isobar_Input(in_para): 

    counter = 0 

    T = {} 
    
    N_tol = {} 

    W = {} 

    for match in in_para.matching: 

        arg = match.split()     
        
        if ( arg[0] == "isobar"): 

            subfolder = arg[1] 

            T_start = arg.index("T") + 1  

            T_end = arg.index("sort") 

            counter = counter + 1 

            #T[str(counter)] = np.array(arg[T_start:T_end]).astype(np.int)   
    
            T[subfolder] = np.array(arg[T_start:T_end]).astype(np.int) 
        
            #N_tol[str(counter)] = int(arg[-1])     
    
            N_tol[subfolder] = int(arg[-1]) 

            #w[str(counter)] =  

            W[subfolder] = float(arg[2]) 

    return N_tol,T,W

def Compute_Isobar_Norm(ref_address,T,natoms): 

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

def Initialize_Isobar_Matching(in_para,ref_address,natoms): 

    N_tol,T,W = Parse_Isobar_Input(in_para) 

    ref_density, ref_density_norm,T_rank_ref = Compute_Isobar_Norm(ref_address,T,natoms)

    return N_tol,T,W,ref_density, ref_density_norm,T_rank_ref 

def Compute_Density(natoms,sub_folders,num_T,real_units):   

    volume = np.zeros(num_T) 

    for indx,address in enumerate(sub_folders):

        volume[indx] = np.average( np.loadtxt(address+"/dump.volume" ) )    

    return  ( natoms/const_mod.NA)*const_mod.water_mol_mass/(volume*real_units.vol_scale*10**6)  

def Isobar_Matching(W,natoms,file_address,UNITS):

    costfunc = 0 

    for subfolder in file_address: 

        list_address = file_address[subfolder] 

        num_T = len(list_address) 
    
        predict_density = Compute_Density(natoms,list_address,num_T,UNITS)  

        indx = np.argsort(predict_density) 

        T_predict_sort = np.array(T[subfolder]).astype(np.int)[indx] 

        diff = predict_density - ref_density[subfolder]

        penalty = Penalty( N_tol_all[subfolder],T_predict_sort,T_rank_ref[subfolder]) 
    
        print (penalty)

        scaled_isobar = np.sum((diff)**2/ref_density_norms[subfolder]) 

        costfunc = costfunc + ( scaled_isobar*penalty + scaled_isobar )*W[subfolder]
            
    return costfunc

#------------------------------------------------------------------------------------------- 
#        RDF Matching  
#--------------------------------------------------------------------------------------------

def wait_gr_dcd_files(dcdfile,output_address): 

    natoms,nconfigs_now = reader.readdcdheader(dcdfile)

    total_nconfigs = np.loadtxt(output_address+"/"+"jobfinish.txt")     
    
    counter = 0 

    while ( int(total_nconfigs) != nconfigs_now  ): 
        
        nconfigs_now = reader.readdcdheader(dcdfile)   

        counter = counter + 1 
    
        if ( counter >= 2000):  

            print ("Wait too long for trajectory" )

            sys.exit() 

    return None     

def infer_num_bins_cutoff(r_lst): 

    num_bins = r_lst.size 
    
    return None 
    
def GetBins(num_bins,cutoff): 
    
    r_interval = cutoff/num_bins

    bins_position = np.zeros(num_bins) 
    
    for i in range(num_bins): 

        bins_position[i] = r_interval*0.5 + i*r_interval 
    
    return bins_position

# Write RDF histogram: 1st column is r ; 2nd column is gr  

def initialize_rdf_matching(ref_address_dict,cutoff,num_bins): 

    reffile = "Ref.gr"

    ref_data = {} ; ref_norm = {}  

    r_dist = GetBins(num_bins,cutoff) 

    r_interval = cutoff/num_bins 

    for subfolder in ref_address_dict: 

        ref_address_lst = ref_address_dict[subfolder]
        
        all_data = [] ; all_ref_norm = []  

        for ref_address in ref_address_lst: 

            gr = np.loadtxt(ref_address  + "/" + reffile)  
            
            gr_ref = np.interp(r_dist,gr[:,0],gr[:,1]) 

            sum_ref = 0.0  

            for i in range(num_bins): 

                sum_ref = sum_ref + r_interval*(r_dist[i]*gr_ref[i] -r_dist[i])**2 
            
            all_ref_norm.append( sum_ref)  
            
            all_data.append(gr)     

        ref_data[subfolder] = all_data 
    
        ref_norm[subfolder] = all_ref_norm 

    return ref_data,ref_norm 

def WriteRDF(filename,gr,radius):  

    np.savetxt(filename,zip(radius,gr))  

# Compute RDF histogram from a dcd trajectory by using serial job 

def ComputeRDF_bySerial(dcdfile,cutoff,num_bins): 

    natoms,total_frames = reader.readdcdheader(dcdfile) 

    total_atoms, total_volume = 0.0,0.0 

    radius = GetBins(num_bins,cutoff) 

    for current_frame in range(total_frames): 

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

    for i in range(num_cores): 

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

    for current_frame in range(start,end+1): 

        xyz,box = reader.read_dcd_xyzbox(dcdfile,natoms,current_frame) 
        
        vol = np.prod(box) 

        rdf_hist = rdf_hist + computerdf.build_homo_pair_distance_histogram(natoms,cutoff,num_bins,xyz,box)

        total_volume = total_volume + vol 
        
        total_atoms = total_atoms + natoms 

    return rdf_hist, total_atoms, total_volume

# RDF Matching functional forms 

def compute_scaled_gr(num_cores,dcdfile,cutoff,num_bins,norm,gr_ref,counter): 

    r_dist = GetBins(num_bins,cutoff) 

    gr_predict = LanuchParallel_gr(num_cores,dcdfile,cutoff,num_bins)

    r_interval = cutoff/num_bins 
    
    WriteRDF(output.output_address + "dump_%d.gr"%(counter),gr_predict,r_dist) 

    sqr_diff = ((r_dist*gr_predict - r_dist*gr_ref))**2  

    sum_rgr_diff  = 0.0 ; sum_ref =0.0 ; test =0.0 

    for index,y_diff in enumerate(sqr_diff) : 

        sum_rgr_diff  = sum_rgr_diff + r_interval*y_diff  
    
    return sum_rgr_diff/(norm) 

# Sum_all_rdf

def compute_gr_matching(num_cores,predict_data_address,num_bins,cutoff): 

    sum_obj = 0 

    counter = 0     

    for subfolder in predict_data_address: 
    
        file_list_address = predict_data_address[subfolder]

        all_ref_gr_data = ref_gr_data[subfolder] 

        all_ref_gr_norm = ref_gr_norm[subfolder] 

        for i,address in enumerate(file_list_address):      

            gr_ref = all_ref_gr_data[i][:,1]
            
            ref_norm = all_ref_gr_norm[i]  

            dcdfile = address + "/"  + "traj.dcd"

            wait_gr_dcd_files(dcdfile,address) 

            counter = counter + 1 

            scaled_gr = compute_scaled_gr(num_cores, dcdfile, cutoff,num_bins,ref_norm,gr_ref,counter)

            sum_obj = sum_obj + scaled_gr   

    return sum_obj  
            
def Compute_force_matching(file_address):

    for subfolder in file_address: 
    
        list_address = file_address[subfolder]
    
        for list_of_file in list_address: 
    
            trajdcd = list_of_file + "/" + "traj.dcd" 

    return None 

#------------------------------------------------------------------------------------------- 
#--------------------------- Isobars Matching ---------------------------------------------- 
#-------------------------------------------------------------------------------------------


#-----------------------------------------------------------------------------
#-------------------------- Force Matching -----------------------------------
#-----------------------------------------------------------------------------

def initialize_force_matching(ref_address_dict):

    energy_file = "Ref.eng" 
 
    force_file = "Ref.force"

    pe_data = {}

    pe_norm = {}    

    force_lines = {} 

    for subfolder in ref_address_dict: 

        ref_address_lst = ref_address_dict[subfolder]
        
        pe_data_lst = [] ; pe_norm_lst = [] ; force_lines_lst = []  

        for ref_address in ref_address_lst: 

            energy_data = ref_address + "/" + energy_file  

            force_data = ref_address + "/" + force_file  

            num_lines = reader.getlines(force_data) 

            force_lines_lst.append( num_lines) 
        
            pe = np.loadtxt(energy_data)

            pe_data_lst.append(pe) 

            pe_norm_lst.append(np.var(pe)) 
    
        pe_data[subfolder] = pe_data_lst   
        
        pe_norm[subfolder] = pe_norm_lst  

        force_lines[subfolder] = force_lines_lst  

    return pe_data,pe_norm, force_lines  

def parse_matching(argument):

    # get weighting argument for each type matching
    
    matching_weight = np.zeros(len(argument)) 

    for index,arg in enumerate(argument): 

        content = arg.split() 

        matching_weight[index] = float(content[2])  

    return matching_weight 

def parse_force_matching_type(argument):  

    arg_lst = argument.split() 
        
    if ( "w" in argument ): 

        start = arg_lst.index("w") + 1 
    
        end = start + 2             

        fm_weight= np.array(arg_lst[start:end]).astype(np.float64)  

    if ( "bf" in argument): 

        index = arg_lst.index("bf") + 1  

        fm_buffersize = np.array(arg_lst[index]).astype(np.int)     

    return fm_weight, fm_buffersize 

def Wait_Force_data(forcefile,Ref_force_lines): 

    counter =  0  

    while ( sys_mod.get_lines(forcefile) != Ref_force_lines): 

        time.sleep(1) 

        counter = counter + 1 

        if ( counter >= 2000 ): 
    
            print ("Force data is not ready 30mins after program finishes") 

            sys.exit() 

    return None 

def Compute_Eng_Normalization(ref_file_address): 

    # Load Reference data 

    eng_ref = np.loadtxt(ref_file_address)  

    # Compute Normalization constant 

    return np.var(eng_ref)  

def ComputeScaledEnergy(eng_ref,eng_para): 

    ave_diff = np.average(eng_para - eng_ref )   

    diff = eng_para - eng_ref 

    relative_eng = ( diff -ave_diff )**2 

    eng_var = np.var(eng_ref) 

    scaled_eng = relative_eng/eng_var 

    return np.average(scaled_eng)  

def Compute_New_Eng(num_atoms,eng_ref,eng_para,eng_norm): 

    return np.average(((eng_para - eng_ref)**2/eng_norm))

def Compute_Force_Normalization(total_configs,chunksize,num_atoms,num_column,Reffile,skip_ref): 

    num_itera = total_configs/chunksize 

    sum_refforce = 0.0

    sqr_ave = 0.0  
    
    datasize = chunksize*( num_atoms+ 9 ) 

    for i in range(num_itera): 
        
        start = i*datasize 

        end = start + datasize 
    
        Ref_chunkdata = ReadFileByChunk(Reffile, start,end) 

        sum_refforce = sum_refforce + np.sum(Ref_chunkdata) 
    
        sqr_ave = sqr_ave + np.sum(Ref_chunkdata*Ref_chunkdata)     

        print (Ref_chunkdata[1:90]) 

    average_sqr = (sum_refforce/(total_configs*num_atoms*num_column))**2

    sqr_average =  sqr_ave/(total_configs*num_atoms*num_column)

    variances_ref = sqr_average - average_sqr

    return variances_ref 

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

def ComputeParallelForceObjective(total_configs,chunksize,num_atoms,num_column,dumpfile,Reffile,skip_ref,skip_dump):
    p = mp.Pool(2) 

    datafile = [ Reffile,dumpfile] 

    num_itera = int(total_configs/chunksize)

    remainder = int(total_configs%chunksize)

    if ( remainder != 0 ):

            print ("Chunksize has to be divisible by total configurations")

            print ("Chunksize is: ", chunksize, " and total configurations are: ", total_configs)

            sys.exit()

    sum_refforce = 0.0 ; sqr_ave = 0.0 ; sum_diff  = 0.0

    datasize = chunksize*( num_atoms+ 9 )

    for i in range(num_itera):

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

def ComputeSumSquared(force_ref_chunk,force_dump_chunk):  

    diff =  force_dump_chunk - force_ref_chunk  

    return np.sum((force_dump_chunk - force_ref_chunk)**2)

def compute_force_matching(W,num_atoms,chunksize,total_configs,file_address,ref_file_add):  

    for subfolder in file_address: 
    
        list_address = file_address[subfolder]
        
        ref_address = ref_file_add[subfolder] 
        
        force_lines_lst = force_lines[subfolder] 

        norm_lst = ref_pe_norm[subfolder]   

        ref_pe_data_lst = ref_pe_data[subfolder]    

        for i,list_of_file in enumerate(list_address): 
    
            energy_file = list_of_file + "/" + "dump.eng" 

            force_file = list_of_file + "/" + "dump.force"

            ref_force = ref_address[i] + "/" + "Ref.force"
    
            num_column = 3 
            
            Ref_force_lines = force_lines_lst[i]    
            
            Wait_Force_data(force_file,Ref_force_lines) 

            scaled_force = ComputeParallelForceObjective(total_configs,chunksize,num_atoms,num_column,force_file,ref_force,skip_ref=0,skip_dump=0)  

            energ_para = np.loadtxt(energy_file) 
    
            #energy_ref = np.loadtxt(Ref_energy) 
            energy_ref = ref_pe_data_lst[i]

            norm = norm_lst[i]

            output.Update_Properties(list_of_file,["dump.eng","dump.force"])

            scaled_energy = Compute_New_Eng(num_atoms,energy_ref,energ_para,norm ) 

            print ("Energy + Force:", fm_weight[0]*scaled_energy, fm_weight[1]*scaled_force) 
            #scaled_energy = ComputeScaledEnergy(energy_ref,energ_para)  
            #print fm_weight[0], fm_weight[1]   
            #print "scaled energy, force", fm_weight[0]*scaled_energy,fm_weight[1]*scaled_force 

    return  fm_weight[0]*scaled_energy + fm_weight[1]*scaled_force

def ComputeObjective_by_Lammps(W,fitting_force_field): 

    costfunclogger = logging.getLogger("In Cost function evalutations: ") 

    # packing fitting parameters into full parameters by adding fix parameters  

    all_force_field_parameters = regroup_parameters(in_para.guess_parameter,in_para.fit_and_fix,fitting_force_field) 

    LAMMPS.Run(all_force_field_parameters) 
    
    job_runs_successful = LAMMPS.Exit() 

    if ( job_runs_successful ): 

        pass 
    
    else:

        print ("LAMMPS Jobs fails; Check error/output/lammps.log/ file ")

        sys.exit()
    
    costfunclogger.debug("Data is ready to be Loaded ... ") 

    num_atoms = 512 ; chunksize = 5000 ; total_configs = 5000  

    scaled_force_obj = compute_force_matching(W,num_atoms,chunksize,total_configs,predict_address["force"],ref_address["force"]) 
    
    #scaled_rdf_obj = compute_gr_matching(num_cores,predict_address["rdf"],num_bins,cutoff) 

    print ("scaled force:", weight_matching[0]*scaled_force_obj) #, "scaled rdf:", weight_matching[1]*scaled_rdf_obj  

    #return weight_matching[0]*scaled_force_obj + weight_matching[1]*scaled_rdf_obj  
    return weight_matching[0]*scaled_force_obj 

def Initialize_virial_matching(temp_file): 

    return None 

def virial_matching_yes(ref_temp_file): 

    temp_data = np.loadtxt(ref_temp_file )   

    matching_index = (temp_data != 0)  

    return matching_index 

def virial_matching(): 


    return None 

def ComputeObjective_by_Lammps_Isobar(W,force_field_parameters):

    costfunclogger = logging.getLogger("In Cost function evalutations: ") 

    LAMMPS.Run(force_field_parameters)

    job_runs_successful = LAMMPS.Exit() 

    costfunclogger.debug("Start Reading output from LAMMPS ... ")   

    if ( job_runs_successful ): 

        #file_ready = isobars_Output_Ready(predict_sub_folders,"dump.volume",num_configs=400) 
        pass

    else:

        print ("LAMMPS Jobs fails; Check errors ")

        sys.exit()

    Isobar_data_Ready(predict_address["isobar"])   

    costfunclogger.debug("Data is ready to be Loaded ... ")     

    natoms = 512

    scaled_isobar = Isobar_Matching(W,natoms,predict_address["isobar"],UNITS)  

    costfunclogger.debug("Finish Reading output from LAMMPS ... ")  

    costfunclogger.info("The scaled isobar is: %.6f"%scaled_isobar )  

    return scaled_isobar 

def PrintNelderMeadRuntime(ordered_para,vertices_sorted): 

    runtimelogger = logging.getLogger("Parameter after NelderMead Simplex: ")  

    runtimelogger.info( "Best vertex: " ) 
    runtimelogger.info( "Parameters: " + " ".join(str(para) for para in ordered_para[0,:]) + "\n" )  
    runtimelogger.info( "Objective function: " +  str(vertices_sorted[0]) + "\n")  
    
    runtimelogger.info( "Worst vertex: ") 
    runtimelogger.info( "Parameters: " + " ".join(str(para) for para in ordered_para[-1,:]) + "\n" )  
    runtimelogger.info( "Objective function: " + str(vertices_sorted[-1]) + "\n" )  

    return None 

def optimization_output(best_para,best_obj):

    with open("optimization_%s_is_done.txt"%str(JOBID),"w") as finish_out:

        finish_out.write(str(best_obj)+ "\n" )

        finish_out.write(" ".join(str(para) for para in best_para) + "\n" )

    return None 

def Nelder_Mead_simplex_converge(n_itera,n_max_itera,vertices_sorted,ordered_para,obj_tol,para_tol,best_para):

    terminate_logger = logging.getLogger(__name__)  

    terminate_logger.debug("function:terminate function entered successfully !") 

    if ( (np.amax(vertices_sorted)/np.amin(vertices_sorted) - 1)< obj_tol ): 

        sci_obj = "{0:.1e}".format(obj_tol) 

        log.info("Convergence criterion 1 is met: Ratio of obj_max/obj_min -1  < %.6f !\n"%sci_obj)

        log.info( "Optimization converges and program exits ! \n")

        optimization_output(best_para,vertices_sorted[0]) 
    
        sys.exit()

        return None 

    unique_obj,repeat = np.unique(vertices_sorted,return_counts=True) 

    if ( unique_obj.size < vertices_sorted.size ):

        log.info("Convergence criterion 2 is met: some objective functions of different vertex begin to converge" )
    
        log.info(" ".join(str(obj) for obj in vertices_sorted) ) 

        log.info( "Optimization converges and program exits ! \n")

        optimization_output(best_para,vertices_sorted[0]) 

        sys.exit()

        return None 
    
    if ( np.all(np.std(ordered_para) < para_tol ) ):    
        
        sci_para = "{0:.1e}".format(para_tol)
    
        log.info("Convergence criterion 3 is met: the standard deviation of force-field paramteters across all vertices is <%.6f  !\n"%sci_para)

        log.info( "Optimization converges and program exits ! \n")

        optimization_output(best_para,vertices_sorted[0])

        sys.exit()

        return None 

    if ( n_itera  == n_max_itera ): 

        log.info("Convergence criterion 4 is met: Maximum number of iteration is reached !\n")

        optimization_output(best_para,vertices_sorted[0])

        sys.exit( "Maximum iteration %d is reached and Program exit !"%n_max_itera)

        return None 

if ( __name__) == "__main__": 

    # ----------------------- Taking input argument --------------------------
    
    # Two modes in reading input argument:      

    # Command line mode: leave the passed argument blank 

    #log,TOTAL_CORES,INPUTFILES,JOBID = sys_mod.ReadCommandLine().Finish()  

    # Interactive mode: explicitly pass the input argument  

    log,TOTAL_CORES,INPUTFILES,JOBID = sys_mod.ReadCommandLine(jobID="Jupeter-Notebook_demo",
                                                               total_cores=2,
                                                               input_file="in.para").Finish()
    
    # Return:                                                                   

    # --log (obj): log object that writes a log file collecting all logging information from different modules during the runtime 
    # global variables:  
    # --TOTAL_CORES (int): total number of cores assigned by slurm before running the program 
    #    a. cores used by simulation engine  ( less than or equal to TOTAL_CORES )   
    #    b. cores used by computing objective function ( less than or equal to TOTAL_CORES ) 
    # --INPUTFILES (str): the name of input file with optimization settings 
    # --JOBID (str): Job ID given by user or slurm job ID; create a unique tag for working folders and log file 

    # ----------------------- Parse Input file -----------------------------------

    # read input file and initialize other object
    
    in_para = sys_mod.ParseInputFile(INPUTFILES) 

    # Return:  

    # --in_para (obj): contains all optimization settings in input file   
     
    # ----------------------- Set up working folders -----------------------------

    HOME,ref_address,predict_address = sys_mod.Setup(JOBID,in_para.matching,overwrite=True).Finish()    
    
    # Return: 
    # --HOME: address of current working directory 
    # --ref_address: folder address containing the reference data  
    # --predict_address: the address where LAMMPS is launched with optimized force-field parameters 
    
    # ----------------------- Initialize simulation engine  ----------------------------

    LAMMPS = simulation.Invoke_LAMMPS(in_para,predict_address,TOTAL_CORES,HOME) 

    # Return:
    # --LAMMPS: the simulation engine chosen 

    # ----------------------- Initialize Output -------------------------------- 
    
    output = sys_mod.Output(in_para,JOBID) 

    # ------------------------ Print the input --------------------- 

    PrintInputSettings(
                       in_para.guess_parameter,
                       in_para.fit_and_fix,
                       in_para.constraints_index,
                       in_para.constraints_bound,
                       in_para.max_iteration,
                       in_para.obj_tol,
                       in_para.para_tol, 
                       )

    # ----------------------- Initialize Cost Function ----------------------------

    W = 1 

    #ref_gr_data,ref_gr_norm = initialize_rdf_matching(ref_address["rdf"],cutoff,num_bins)

    ref_pe_data,ref_pe_norm,force_lines = initialize_force_matching(ref_address["force"])
    
    fm_weight,fm_buffer = parse_force_matching_type(in_para.matching[0]) 

    weight_matching = parse_matching(in_para.matching) 

    #-------------------------------------------------------

    NVertices,vertices_sorted,ordered_para = InitializeNelderMeadSimplex(W,in_para) 

    alpha,kai,gamma,sigma = SelectNelderMeadParameters("adaptive",NVertices-1)  

    log.info("Optimization begins ... \n ") 
    
    for itera in range(in_para.max_iteration+1):

        worst = vertices_sorted.size -1  
        
        second_worst = vertices_sorted.size-2  

        best = 0  

        except_best = best + 1  

        output.Write_Best_Worst_Vertices(itera,vertices_sorted[best],vertices_sorted[worst]) 

        best_para = regroup_parameters(in_para.guess_parameter,in_para.fit_and_fix,ordered_para[best,:]) 

        output.Write_Best_Parameters(best_para)

        output.Write_Restart(itera,ordered_para[best,:],ordered_para,vertices_sorted) 

        Nelder_Mead_simplex_converge(itera,in_para.max_iteration,vertices_sorted,ordered_para,in_para.obj_tol,in_para.para_tol,best_para)   

        log.info("\n") 
        log.info("-------------------------------------------Current iteration-------------------------------------------\n") 
        log.info("                                                 %d                                                    " %itera) 
        log.info("-------------------------------------------------------------------------------------------------------\n")  
        log.info("Compute the centroid of all vertices except the worst vertex ...\n") 
        log.info("Reflect the worst (with the highest objective function) vertex accross the centroid ... \n") 

        # compute centroid  

        centroid = ComputeCentroid(ordered_para,worst)

        # Reflection Operation: Relect the worst vertices
        
        vr_x = NelderMeadReflection(centroid,ordered_para[worst,:])
        
        # get objective function values at reflected vertices 

        fvr_x = ComputeObjective_by_Lammps(W,vr_x)   

        if ( fvr_x < vertices_sorted[second_worst] and vertices_sorted[best] <= fvr_x):  
            
            ordered_para[worst,:] = vr_x 

            vertices_sorted[worst] = fvr_x      

            log.info("Since the objective function of reflected vertex  is worst (higher) than best (lowest) vertex but better than that of second worst (second highest) vertex ... \n ") 
            log.info("The reflected vertex is accpeted and replace the worst vertex ...\n" )
            log.info( "Now sort all vertices values... \n" )  
            
            vertices_sorted,ordered_para = OrderVertices(vertices_sorted,ordered_para)

            PrintNelderMeadRuntime(ordered_para,vertices_sorted) 

            log.info("Start new iteration ... \n") 

            #simplex.vertices =ordered_para ; simplex.cost = vertices_sorted 
            
            #WriteRestart(currentrestart,itera,restartfreq,ordered_para[best,:],fit_and_unfit,set_parameter[2],simplex) 

            #WriteRestartLog(logrestart,restartfreq,itera,ordered_para[best,:],fit_and_unfit,set_parameter[2],simplex)  

            continue        

        # Expansion 
        
        if ( fvr_x < vertices_sorted[best]): 
            
            log.info( "The objective function value of reflected vertex is better (lower) than the best vertex ... \n" ) 

            log.info( "Then, Further explore the reflected direction by expansion from the current reflected vertex ... \n ") 

            ve_x = NelderMeadExpansion(centroid,vr_x) 

            feve_x = ComputeObjective_by_Lammps(W,ve_x)

            if (  feve_x < fvr_x ): 
            
                log.info( "The objective function value of expanded vertex is better (lower) than value of reflected vertex ... \n" )  
        
                ordered_para[worst,:] = ve_x  

                vertices_sorted[worst] = feve_x  

                log.info( "The expanded vertex is accpeted and replace the worst vertex ... \n" )  
            
                log.info( "Now sort all vertices values ... \n")  
            
                vertices_sorted,ordered_para = OrderVertices(vertices_sorted,ordered_para) 

                PrintNelderMeadRuntime(ordered_para,vertices_sorted)

                log.info( "Start new iteration ... \n")  

                #simplex.vertices =ordered_para ; simplex.cost = vertices_sorted 
                
                #WriteRestart(currentrestart,itera,restartfreq,ordered_para[best,:],fit_and_unfit,set_parameter[2],simplex) 

                #WriteRestartLog(logrestart,restartfreq,itera,ordered_para[best,:],fit_and_unfit,set_parameter[2],simplex)  

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

                #simplex.vertices =ordered_para ; simplex.cost = vertices_sorted 
                
                #WriteRestart(currentrestart,itera,restartfreq,ordered_para[best,:],fit_and_unfit,set_parameter[2],simplex) 

                #WriteRestartLog(logrestart,restartfreq,itera,ordered_para[best,:],fit_and_unfit,set_parameter[2],simplex)  

                continue    

        # Contraction: 
        
        if ( fvr_x  >= vertices_sorted[second_worst] ): 

            log.info( "Reflected vertex (objective function) is worst (higher) than that of second-worst vertex ... \n") 

            if ( fvr_x < vertices_sorted[worst] ): 
                
                log.info( "Reflected vertex is better than that of worst vertex ... \n") 
                
                log.info( "So, keep the reflected vertex first and perform the outside contraction to shrink the reflected value toward the centroid ... \n" ) 
                
                # outside contraction   
                
                v_oc = NelderMeadContractionO(centroid,vr_x)
                
                fc_outside = ComputeObjective_by_Lammps(W,v_oc)
                
                if ( fc_outside <= fvr_x ):

                    log.info( "After the outside contraction, the resulting objective function of new vertex is better (lower) than that of reflected vertex ... \n" ) 
                    
                    log.info( "Then, the new contracted vertex is accpeted and replace the worst vertex ... \n") 
                    
                    log.info( "Now sort all vertices values ... \n")  

                    ordered_para[worst,:] = v_oc 

                    vertices_sorted[worst] = fc_outside 

                    vertices_sorted,ordered_para = OrderVertices(vertices_sorted,ordered_para) 

                    PrintNelderMeadRuntime(ordered_para,vertices_sorted) 

                    log.info( "Start new iteration ... \n")  

                    #simplex.vertices =ordered_para ; simplex.cost = vertices_sorted 
                    
                    #WriteRestart(currentrestart,itera,restartfreq,ordered_para[best,:],fit_and_unfit,set_parameter[2],simplex) 

                    #WriteRestartLog(logrestart,restartfreq,itera,ordered_para[best,:],fit_and_unfit,set_parameter[2],simplex)  

                    continue 

                else:  

                    log.info("After the outside contraction, the resulting objective function of new contracted vertex is worse (higher) than that of reflected vertex ... \n")  
                    
                    log.info( "Then, the shrinkage will are performed ... \n")   

                    x_shrink = NelderMeadShrinkVertices(ordered_para,NVertices)
                    
                    vertices_shrinked = ComputeFuncVertices(W,x_shrink) 

                    vertices_sorted[1:] = vertices_shrinked     

                    ordered_para[1:,:] = x_shrink 

                    vertices_sorted,ordered_para = OrderVertices(vertices_sorted,ordered_para) 

                    log.info( "Now sort all vertices values ... \n")  

                    PrintNelderMeadRuntime(ordered_para,vertices_sorted) 

                    log.info( "Start new iteration ... \n")   

                    #simplex.vertices =ordered_para ; simplex.cost = vertices_sorted 
                    
                    #WriteRestart(currentrestart,itera,restartfreq,ordered_para[best,:],fit_and_unfit,set_parameter[2],simplex) 

                    #WriteRestartLog(logrestart,restartfreq,itera,ordered_para[best,:],fit_and_unfit,set_parameter[2],simplex)  

                    continue 

            elif ( fvr_x >= vertices_sorted[worst]):    
                
                # inside contraction 

                log.info( "Reflected vertex is even worse (higher) than that of the worst vertex ...  \n")  
                
                log.info( "Then, reject the reflected vertex and perform the inside contraction to bring the worst vertex toward the centroid ... \n" )   
            
                v_ic = NelderMeadContractionI(centroid,ordered_para[worst,:])   

                fc_inside = ComputeObjective_by_Lammps(W,v_ic)

                if ( fc_inside < vertices_sorted[worst] ): 

                    log.info( "Now, the objective function of new vertex is better than that of the worst vertex ... \n") 
                    
                    log.info( "Then, accept the new contracted vertex and replace the worst vertex with it ... \n")  

                    ordered_para[worst,:] = v_ic

                    vertices_sorted[worst] = fc_inside

                    vertices_sorted,ordered_para = OrderVertices(vertices_sorted,ordered_para) 

                    log.info( "Now sort all vertices values ... \n")  

                    PrintNelderMeadRuntime(ordered_para,vertices_sorted) 

                    log.info( "Start new iteration ... \n")  

                    #simplex.vertices =ordered_para ; simplex.cost = vertices_sorted 
                    
                    #WriteRestart(currentrestart,itera,restartfreq,ordered_para[best,:],fit_and_unfit,set_parameter[2],simplex) 

                    #WriteRestartLog(logrestart,restartfreq,itera,ordered_para[best,:],fit_and_unfit,set_parameter[2],simplex)  

                    continue

                else:   
                    
                    log.info( "Now, the objective function of new vertex is still worst than than that of the worst vertex ... \n") 
                    
                    log.info( "Then, the shrinkage are performed ... \n")   

                    x_shrink = NelderMeadShrinkVertices(ordered_para,NVertices)  
    
                    vertices_shrinked = ComputeFuncVertices(W,x_shrink) 

                    vertices_sorted[except_best:] = vertices_shrinked

                    ordered_para[except_best:,:] = x_shrink 

                    vertices_sorted,ordered_para = OrderVertices(vertices_sorted,ordered_para) 
                    
                    log.info( "After shrinkage, now sort all vertices values ... \n")  
                    
                    PrintNelderMeadRuntime(ordered_para,vertices_sorted) 

                    log.info( "Start new iteration ... \n")  

                    #simplex.vertices =ordered_para ; simplex.cost = vertices_sorted 
                    
                    #WriteRestart(currentrestart,itera,restartfreq,ordered_para[best,:],fit_and_unfit,set_parameter[2],simplex) 

                    #WriteRestartLog(logrestart,restartfreq,itera,ordered_para[best,:],fit_and_unfit,set_parameter[2],simplex)  

                    continue 


