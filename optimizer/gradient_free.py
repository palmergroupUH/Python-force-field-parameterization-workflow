import numpy as np 
import logging 
import sys 
import optimizer.optimizer_mod
import random 

# a gradient-free optimizer: Nelder-Mead simplex metho:  

class NelderMeadSimplex(optimizer.optimizer_mod.set_optimizer):
    
    def __init__(self,
                input_files,
                logname, 
                f_objective,  
                skipped=None, 
                Output=None,
                optimize_mode=None,
                nm_type=None):

        
        # Inherit the following from parent class: set_optimizer

        super().__init__(input_files,logname,skipped)

        # Fileds Inherited: 

        # 1. guess parameters (self.guess_parameter)   
        # 2. parameters to be fitted or fixed ( self.fit_and_fix )  
        # 3. index of guess parameters to be constrained ( self.constraints_fit_index ) 
        # 4. constraints bounds (self.constraints_bounds )  
        # 5. mode of Nelder-Mead simplex: "Perturb" or "Restart" ( self. optimizer )  
        # 6. contents of optimizer ( self.optimizer_input )  

        # Methods Inherited: 

        # 1. self.constrain() 
        # 2. self.regroup_parameter() 


        # computing objective function 

        self.f_obj = f_objective
    
        if ( optimize_mode is None ):  
        
            self.optimize_mode = "minimize" 

        else: 

            self.optimize_mode = optimize_mode 

        # Define the following instance variables: 

        # self.vertices_sorted: all vertices that have been sorted  
        # self.func_vertices_sorted: all function values sorted at vertices 
        # self.num_vertices: number of vertices
        # self.worst: 
        # self.best 
        # self.lousy 

        # check optimizer type and its mode:     

        self.parse_Nelder_Mead_Input() 

        # initialize simplex 

        self.initialize_simplex()

        # default: adaptive nelder-mead simplex coefficient 
        # check out  

        self.TransformationCoeff("standard")

    # check the general input:  

    def parse_Nelder_Mead_Input(self):

        if ( self.optimizer_type == "Nelder-Mead"):  

            # at least 1 argument needed for Nelder-Mead: either restart or perturb

            if (len(self.optimizer_argument) == 1 ): 

                self.optimizer_logger.error("ERROR: Missing a mode argument: 'Peturb' or 'Restart'")
    
                sys.exit("Check errors in log file !") 

            self.optimizer_mode = self.optimizer_argument[1]   
            
        else: 
            
            self.optimizer_logger.error("Optimizer: %s not recognized ! Please choose optimizer: 'Nelder-Mead'  "%self.optimizer_type )

            sys.exit("Check errors in log file !")

    def check_Nelder_Mead_mode(self): 

        # Check each Nelder-Mead mode: 

        if ( self.optimizer_argument[1] != "Perturb"
         and self.optimizer_argument[1] != "Restart"):   

            self.optimizer_logger.error("ERROR: only two modes allowed in Nelder-Mead Simplex: Perturb or Restart")
            
            sys.exit("Check errors in log file !") 

        # Extract Perturb argument:  

        # if Perturb used and at least two arguments are provided: 

        if ( self.optimizer_argument[1] == "Perturb" 

            and len(self.optimizer_argument) == 4 ): 

            optimizer_mode_arg = self.optimizer_argument[2:] 

            self.check_Nelder_Mead_perturb_mode(optimizer_mode_arg)

        elif ( self.optimizer_argument[1] == "Perturb" 

            and len(self.optimizer_argument) == 2 ): 

            self.check_provided_perturb_stepsize()  
           
        else:

            self.optimizer_logger.error("ERROR: invalid Nelder-Mead Perturb mode and arguments") 
        
            sys.exit("Check errors in log file !")
    
        # Extract Restart argument 

        if ( self.optimizer_argument[1] == "Restart" ):  

            self.check_restart_argument() 

        return None  

    def check_Nelder_Mead_perturb_mode(self,optimizer_mode_arg):     

        # check 1st argument: 

        if ( optimizer_mode_arg[0] != "random"
             and optimizer_mode_arg[0] != "+"
             and optimizer_mode_arg[0] != "-"  ): 

            self.optimizer_logger.error("ERROR: If the 'Perturb' mode is used, its argument can only be: 'random', '+', '-'\n"
                                         "The mode arugment: %s found in the input file"%optimizer_mode_arg[0] )
            
            sys.exit("Check errors in log file !") 

        else:

            self.NM_perturb_mode = optimizer_mode_arg[0]
        
        # check 2nd argument: 

        try:

            self.NM_perturb_stepsize = float(optimizer_mode_arg[1])  

        except ( ValueError, TypeError ): 

            self.optimizer_logger.error("ERROR: When Nelder-Mead 'Perturb' mode is used"
                                        ",and arguments are provided "
                                        "; The second argument should be float ( percentage)")
             
            sys.exit("Check errors in log file !")

        return None 

    def check_provided_perturb_stepsize(self):

        if ( len(self.optimizer_input)  !=  1 ):   

            self.optimizer_logger.error("ERROR: If Nelder-Mead 'Perturb' mode is used,and no Perturb arguments provided,\n" 
                                        "then,1 row of stepsize ( 0.1, -0.2, 0.8 -0.3 ... ) should be provided by user\n"
                                        "%d rows found in the input file"%len(self.optimizer_input))     
                                        

            sys.exit("Check errors in log file !")

        try: 

            self.stepsize = np.array(self.optimizer_input[0]).astype(np.float64) 

        except  ( ValueError, TypeError ): 

            self.optimizer_logger.error(
                       "ERROR: Invalide perturbed stepsize encountered"
                       " when using Nelder-Mead 'Perturb' mode\n")  
       
            sys.exit("Check errors in log file !") 
        
        return None 

    def generate_simplex_stepsize(self): 
        
        if ( hasattr(self,"NM_perturb_mode") 
         and hasattr(self,"NM_perturb_stepsize")): 
          
            num_fitted = np.sum(self.fit_and_fix==1)  

            self.stepsize = np.zeros(num_fitted) 

            for i in range(num_fitted):  
               
                self.stepsize[i] = self.NM_perturb_stepsize*self.generate_perturb_sign(self.NM_perturb_mode) 

        else:  
    
            self.stepsize = np.array(self.optimizer_input[0]).astype(np.float64) 

            # the number of perturbed vertices must be equal to the number of fitted parameters 

            if ( self.stepsize.size != np.sum(self.fit_and_fix==1) ): 

                self.optimizer_logger.error(
                            "ERROR: When Nelder-Mead 'Perturb' mode is used;" 
                            "The nubmer of perturbed stepsize" 
                            "must be equal to" 
                            "the nubmer of fitted parameters ( = 1 )")

                sys.exit("Check errors in log file !") 

        return None 

    def generate_perturb_sign(self,mode):

        if ( mode == "random"): 

            if ( random.uniform(0,1) < 0.5 ):   

                return 1 
            
            else:
    
                return -1 

        elif ( mode == "+" ):
    
            return 1  

        elif ( mode =="-"): 

            return -1 
             
        else: 

            self.optimizer_logger.error("perturb sign mode not recgonized "
                                       "! Please choose '+','-', or 'random'")
        
            sys.exit("Check errors in log file !")

    def check_restart_argument(self): 

        if ( len(self.optimizer_input)  < 3 ): 

            self.optimizer_logger.error("ERROR: When Nelder-Mead Restart mode is used \n"
                                        "At least 3 rows of arguments: \n" 
                                        "( 1st row: objective functions, 2nd row: first vertex, 3rd row: second vertex ... \n"
                                        "%d rows found in the input file"%len(self.optimizer_input) ) 
       
            sys.exit("Check errors in log file !")  

        # check size of objective functions and vertices provided : 

        number_vertices = len(self.optimizer_input[0]) 
    
        if ( len(self.optimizer_input[1:]) != number_vertices ) : 

            self.optimizer_logger.error("ERROR: When Nelder-Mead Restart mode is used: \n" 
                                        "Number of vertices should be equal to number of vertex parameter") 
            
            sys.exit("Check errors in log file !") 

        # check consistency of parameters with number of vertices

        for i in range(len(self.optimizer_input[1:])):         

            if ( len(self.optimizer_input[i+1]) == number_vertices - 1 ):  

               self.optimizer_logger.error("ERROR: When Nelder-Mead Restart mode is used: \n" 
                                           "Number parameters should be ( number_vertices - 1 )") 

               sys.exit("Check errors in log file !")
        
        return None 

    def parse_existing_simplex(self): 
        
        try: 

            func_vertices = np.array(self.optimizer_input[0] ).astype(np.float64) 

            self.num_vertices = self.func_vertices.size 

            self.num_fitting = self.num_vertices - 1  

            vertices_mat = np.zeros((self.num_vertices,self.num_fitting)) 

            for i in range(len(self.optimizer_input[1:])):

                vertices_mat[i,:] = np.array(self.optimizer_input[i+1:] ).astype(np.float64)

            self.sort_simplex(func_vertices,vertices_mat) 

        except ( ValueError, TypeError ): 

            self.optimizer_logger.error("ERROR: When Nelder-Mead Restart mode is used: "                              
                                        "ValueError or TypeError encountered in reading the restart simplex") 

            sys.exit("Check errors in log file !")

        return None 

    def initialize_simplex(self):     

        # choose the two modes: 

        self.check_Nelder_Mead_mode()  

        # "Perturb" create the new simplex  

        if ( self.optimizer_mode == "Perturb"): 
        
            # either read or create step size based on the input 

            self.generate_simplex_stepsize()

            vertices_mat = self.generate_simplex("orthogonal")
            
            func_vertices = self.compute_func_vertices(vertices_mat)
    
            self.sort_simplex(func_vertices,vertices_mat) 
            
        # "Restart" uses the existing simplex

        elif ( self.optimizer_mode == "Restart"):  
                
            self.parse_existing_simplex()     

            self.sort_simplex(self.func_vertices_sorted,self.vertices_sorted) 

        return None 

    def generate_simplex(self,simplex_type): 
            
        # "1" is the fitted variable               

        self.gussed_fitting_para = self.guess_parameter[self.fit_and_fix==1]

        self.num_fitting = self.gussed_fitting_para.size

        # "0" is the fixed variable  

        self.fix_variable = self.guess_parameter[self.fit_and_fix==0] 
    
        self.num_fixed = self.fix_variable.size

        # Number of vertices: n + 1 ( n is the number of fitting parameters ) 

        self.num_vertices = self.num_fitting + 1 

        # generate orthogonal simplex  
        
        if ( simplex_type == "orthogonal"):

            vertices_mat = self.use_orthogonal_simplex()       
 
        return vertices_mat 

    def use_orthogonal_simplex(self):

        # initialize an array: "vertices_mat" to save all vertices 
        # matrix Format: 
        # vertex 1: [ para1, para2 ... para n ]
        # vertex 2: [ para1, para2 ... para n  ]
        # ...
        # vertex n+1: [ para1, para2 ... para n  ]
        # So vertices_mat has dimension of ( n+1, n )  

        vertices_mat = np.zeros((self.num_vertices,self.num_fitting)) 
             
        # first vertex is the guess parameter     

        vertices_mat[0,:] = self.gussed_fitting_para 
 
        # orthogonal perturbation of vertices

        shift_vector = np.eye(self.num_fitting) 

        # loop over vertex except first one which is the guess parameter  

        for i in range(1,self.num_fitting+1):

            if ( self.gussed_fitting_para[i-1] == 0 ): 

                new_vertices = self.gussed_fitting_para + self.stepsize[i-1]*shift_vector[i-1,:]*0.05 

            else: 

                new_vertices = self.gussed_fitting_para + self.stepsize[i-1]*shift_vector[i-1,:]*self.gussed_fitting_para[i-1]

            self.constrain(new_vertices)

            vertices_mat[i,:] = new_vertices 

        return vertices_mat   

    def sort_simplex(self,func_vertices,vertices_mat):

        # For minimization problems: 
        # sort the objective function from small to large 
        # So, best_objective function is the minima of all objective function 

        if ( self.optimize_mode == "minimize"):  

            self.best_indx = 0 

            self.worst_indx = -1  

            self.lousy_indx = -2 
    
        elif ( self.optimize_mode == "maximize"): 

            self.best_indx = -1              
    
            self.lousy_indx = 1 

            self.worst_indx = 0  

        else: 

            self.optimizer_logger.error("ERROR: optimize mode can only be either 'minimize' or 'maximize'")
    
            sys.exit("Check errors in log file !")

        # argsort default sort order is the asscending order 

        ascending =  np.argsort(func_vertices) 
        
        self.vertices_sorted  = vertices_mat[ascending,:] 

        self.func_vertices_sorted = func_vertices[ascending] 

        # best,worst and lousy objective function values 

        self.best = self.func_vertices_sorted[self.best_indx] 

        self.worst = self.func_vertices_sorted[self.worst_indx] 

        self.lousy = self.func_vertices_sorted[self.lousy_indx] 

        # best,worst and lousy vertex  

        self.best_vertex = self.vertices_sorted[self.best_indx,:]

        self.worst_vertex = self.vertices_sorted[self.worst_indx,:]

        self.lousy_vertex = self.vertices_sorted[self.lousy_indx,:]
        
        return None  

    def TransformationCoeff(self,keyword):  

        if ( keyword == "standard" ): 

            self.alpha = 1 
        
            self.kai =  2.0 

            self.gamma = 0.5 

            self.sigma = 0.5 

        elif ( keyword == "adaptive" ): 

            self.alpha = 1  

            self.kai =  1 + 2.0/self.num_fitting  
    
            self.gamma = 0.75 - 1.0/(2*self.num_fitting)  

            self.sigma = 1.0 - 1.0/self.num_fitting 

    def compute_func_vertices(self,vertices_mat): 

        num_vertices = vertices_mat[:,0].size

        func_vertices = np.zeros(num_vertices) 

        for i in range(num_vertices): 
    
            in_parameters = vertices_mat[i,:]
        
            self.constrain(in_parameters)   

            in_parameters_full = self.regroup_with_fixed(in_parameters) 

            func_vertices[i] = self.f_obj.optimize(in_parameters_full)
        
        return func_vertices   

    def Centroid(self):  
    
        # select all vertices except the worst vertex 

        except_worst = self.vertices_sorted[:self.worst_indx,:]  

        # compute the geometric center 

        return np.mean(except_worst,axis=0)
            
    def Reflect(self,centroid): 
        
        reflected_vetertex = centroid + self.alpha*(centroid - self.worst_vertex) 
       
        self.constrain(reflected_vetertex) 

        return self.regroup_with_fixed(reflected_vetertex) 
 
    def Accept(self,vertex,func_vertex):  

        # subsitude worst vertex 

        self.vertices_sorted[self.worst_indx,:] = vertex  

        self.func_vertices_sorted[self.worst_indx] = func_vertex 

    def Expand(self,reflected,centroid):  

        expanded_vertex = centroid + self.kai*(reflected - centroid ) 

        self.constrain(expanded_vertex)

        return self.regroup_with_fixed(expanded_vertex)  

    def Outside_Contract(self,centroid,reflected_vertex):  

        outside_vertex = centroid + self.gamma*(reflected_vertex - centroid) 

        self.constrain(outside_vertex) 

        return self.regroup_with_fixed(outside_vertex)  
        
    def Inside_Contract(self,centroid):    

        inside_vertex = centroid + self.gamma*( self.worst_vertex - centroid) 

        self.constrain(inside_vertex) 

        return self.regroup_with_fixed(inside_vertex)   
    
    def Shrink(self):   

        shrinked_vertices = np.zeros(( self.num_vertices-1, self.num_vertices-1))

        for i in range(self.num_vertices -1): 
    
            shrinked_vertex = self.best_vertex + self.sigma*(self.vertices_sorted[i+1,:]- self.best_vertex)   

            self.constrain(shrinked_vertex) 

            shrinked_vertices[i,:] = shrinked_vertex 

        func_vertices = self.compute_func_vertices(shrinked_vertices) 
    
        self.vertices_sorted[self.best_indx+1:,:] = shrinked_vertices  

        self.func_vertices_sorted[self.best_indx+1:] = func_vertices 

        self.sort_simplex(self.func_vertices_sorted,self.vertices_sorted)  

        return 

    def print_vertices(self,itera,every):

        if ( itera%every == 0 ): 

            with open("vertices_%d.txt"%itera,"w") as f: 

                np.savetxt(f,np.c_[ [each_vertex for each_vertex in self.vertices_sorted]]) 
                np.savetxt(f,np.c_[ [ self.vertices_sorted[0,:] ]] ) 
        
    def run_optimization(self):  

        # optimization procedures: 
        
        for itera in range(self.max_iteration):  
        
            # Ordering
            # Ascending ( minimization of objective )   
            # Descending ( maximization of objective ) 

            self.print_vertices(itera,every=2) 
           
            if ( self.termination_criterion_is_met(itera)): 

                print ( "current iteration: %d" %itera) 
                print ("best objective: ",self.best) 
                print ( "best parameters:",self.vertices_sorted[self.best_indx,:] ) 

                break 

            # Centroid 

            centroid = self.Centroid() 
            
            # Reflection 
            
            reflected_vertex = self.Reflect(centroid) 
            
            func_reflect = self.f_obj.optimize(reflected_vertex) 
            
            if ( self.best <= func_reflect < self.lousy ): 
                
                self.Accept(reflected_vertex,func_reflect)   
                
                self.sort_simplex(self.func_vertices_sorted,self.vertices_sorted) 
                
                continue 
            
            # Expansion
            
            if ( func_reflect < self.best ): 

                expanded_vertex = self.Expand(reflected_vertex,centroid)          

                func_expand = self.f_obj.optimize(expanded_vertex)

                if ( func_expand < func_reflect ):      

                    self.Accept(expanded_vertex,func_expand) 
                    
                    self.sort_simplex(self.func_vertices_sorted,self.vertices_sorted) 

                    continue 

                else:
                    
                    self.Accept(reflected_vertex,func_reflect ) 
                   
                    self.sort_simplex(self.func_vertices_sorted,self.vertices_sorted) 

                    continue 
            
            # Contraction   

            # outside contraction:  
            
            if ( func_reflect >= self.lousy ):   

                if ( self.lousy <= func_reflect < self.worst ):  
               
                    outside_contract_vertex = self.Outside_Contract(centroid,reflected_vertex) 

                    func_out_contract = self.f_obj.optimize(outside_contract_vertex)

                    if ( func_out_contract <= func_reflect ): 
            
                        self.Accept(outside_contract_vertex,func_out_contract) 

                        self.sort_simplex(self.func_vertices_sorted,self.vertices_sorted) 

                        continue 

                    else:

                        self.Shrink() 
    
                        continue 

                # inside contraction:

                if ( func_reflect >= self.worst ): 

                    inside_contract_vertex = self.Inside_Contract(centroid) 
    
                    func_inside_contract = self.f_obj.optimize(inside_contract_vertex)

                    if ( func_inside_contract < self.worst ):     
                        
                        self.Accept(inside_contract_vertex,func_inside_contract) 

                        self.sort_simplex(self.func_vertices_sorted,self.vertices_sorted) 
        
                        continue 
                
                    else:               
                        
                        self.Shrink()

                        continue    
            
                    
    def termination_criterion_is_met(self,n_itera): 

        self.optimizer_logger.debug("Class NelderMeadSimplex:terminate function entered successfully !")
    
        if ( np.amin(self.func_vertices_sorted) == 0 ):  

            self.optimizer_logger.info("Convergence criterion 1 is met: minimum of objective function is equal to 0")
     
            self.optimizer_logger.info( "Optimization converges and program exits ! \n") 

            return True 

        if ( ( np.amax(self.func_vertices_sorted)/np.amin(self.func_vertices_sorted)-1) < self.obj_tol and divisisor !=0 ):
            
            sci_obj = "{0:.1e}".format(self.obj_tol) 
            
            self.optimizer_logger.info("Convergence criterion 2 is met: Ratio of obj_max/obj_min -1  < %s !\n"%sci_obj) 
        
            self.optimizer_logger.info( "Optimization converges and program exits ! \n") 

            #optimization_output(best_para,self.func_vertices_sorted[0])             

            return True  
    
        unique_obj,repeat = np.unique(self.func_vertices_sorted,return_counts=True) 

        if ( unique_obj.size < self.func_vertices_sorted.size ):
            
            self.optimizer_logger.info("Convergence criterion 3 is met: some objective functions of different vertex begin to converge" )

            self.optimizer_logger.info(" ".join(str(obj) for obj in self.vertices_sorted) )

            self.optimizer_logger.info( "Optimization converges and program exits ! \n")

            #optimization_output(best_para,self.func_vertices_sorted[0])

            return True  

        if ( np.all(np.std(self.func_vertices_sorted) < self.para_tol ) ):
           
            sci_para = "{0:.1e}".format(self.para_tol)

            self.optimizer_logger.info("Convergence criterion 4 is met: the standard deviation of force-field paramteters across all vertices is %s  !\n"%sci_para)

            self.optimizer_logger.info( "Optimization converges and program exits ! \n")

            #optimization_output(best_para,self.func_vertices_sorted[0])

            return True  

        if ( n_itera  == self.max_iteration ): 

            self.optimizer_logger.info("Convergence criterion 5 is met: Maximum number of iteration is reached !\n") 

            #optimization_output(best_para,self.func_vertices_sorted[0]) 

            self.optimizer_logger.info( "Maximum iteration %d is reached and Program exit !"%self.max_iteration)

            return True 

        self.optimizer_logger.debug("Class NelderMeadSimplex:terminate function exit successfully !")
            
        return None 

