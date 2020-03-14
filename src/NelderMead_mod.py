import numpy as np 
import logging 
import sys 


# a gradient-free optimizer: Nelder-Mead simplex metho:  

class NelderMeadSimplex:
    
    def __init__(self,
                in_para,
                f_objective,  
                Output=None,
                mode=None,
                nm_type=None):
    
        # set up the logger  

        self.optimze_logger = logging.getLogger() 

        # "in_para" object contains: 

        # 1. guess parameters  
        # 2. parameters to be fitted or fixed  
        # 3. index of guess parameters to be constrained
        # 4. constraints bounds 
        # 5. mode of Nelder-Mead simplex: "Perturb" or "Restart" 
        # 6. options of selected mode   

        self.in_para = in_para 

        # computing objective function 

        self.f_obj = f_objective

        self.mode = "minimize" 

        # Define the following instance variables: 
        # self.vertices_sorted: all vertices that have been sorted  
        # self.func_vertices_sorted: all function values sorted at vertices 
        # self.num_vertices: number of vertices
        # self.worst: 
        # self.best 
        # self.lousy 

        # initialize simplex 

        self.initialize_simplex()

        # default: adaptive nelder-mead simplex coefficient 
        # check out  

        self.TransformationCoeff("standard")
    
    def initialize_simplex(self):     

        # choose the two modes: 

        # "Perturb" create the new simplex  

        if ( self.in_para.mode == "Perturb"): 

            vertices_mat = self.generate_simplex("orthogonal")

            func_vertices = self.compute_func_vertices(vertices_mat)
    
            self.sort_simplex(func_vertices,vertices_mat) 
            
        # "Restart" uses the existing simplex

        elif ( self.in_para.mode == "Restart"):  
                
            self.use_existing_simplex() 
                
            self.sort_simplex(self.func_vertices_sorted,self.vertices_sorted) 

        return None 

    def generate_simplex(self,simplex_type): 
            
        # "1" is the fitted variable               

        self.gussed_fitting_para = self.in_para.guess_parameter[self.in_para.fit_and_unfit==1]

        self.num_fitting = self.gussed_fitting_para.size

        # "0" is the fixed variable  

        self.fix_variable = self.in_para.guess_parameter[self.in_para.fit_and_unfit==0] 
    
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

                new_vertices = self.gussed_fitting_para + self.in_para.stepsize[i-1]*shift_vector[i-1,:]*0.05 

            else: 

                new_vertices = self.gussed_fitting_para + self.in_para.stepsize[i-1]*shift_vector[i-1,:]*self.gussed_fitting_para[i-1]

            self.constrain(new_vertices)

            vertices_mat[i,:] = new_vertices 

        return vertices_mat   

    def constrain(self,array): 
        
        num_constraints = self.in_para.constraints_fit_index.size 

        num_criterion = np.size(self.in_para.constraints_bound,0) 
        
        if ( num_constraints == num_criterion and num_constraints > 0 ): 

            for i in range(num_criterion): 

                lower = self.in_para.constraints_bound[i][0] 
            
                upper = self.in_para.constraints_bound[i][1]

                constraints_lower_expr  = lower + "<=" + str(array[elf.in_para.constraints_fit_index[i]])

                constraints_upper_expr  =  str(array[self.in_para.constraints_fit_index[i]]) + "<=" + upper

                # evaluate the expression: lower bound < para

                if ( eval(constraints_lower_expr)):

                    # lower bound is indeed < para 

                    pass

                else:

                    self.optimze_logger.info( "Lower constraints are applied...")
                    self.optimze_logger.info( "Parameter: "     
                                              + str( array[self.in_para.constraints_fit_index[i]]) 
                                              + "  is constrained to " + str( lower))

                    array[self.in_para.constraints_fit_index[i]] = lower
        
                # evaluate the expression: lower bound < para

                if ( eval(constraints_upper_expr)):

                    # lower bound is indeed < para

                    pass

                else:

                    self.optimze_logger.info( "Upper constraints are applied..." )

                    self.optimze_logger.info( "Parameter: " 
                                              + str( array[self.in_para.constraints_fit_index[i]]) 
                                              + "  is constrained to " + str( upper))

                    array[self.in_para.constraints_fit_index[i]] = upper

        return None

    def use_existing_simplex(self):
   
        self.func_vertices_sorted = np.array(in_para.obj).astype(np.float64) 
   
        self.vertices_sorted = np.array(in_para.vertices).astype(np.float64)  
    
        self.num_vertices = self.func_vertices_sorted.size
        
        return  

    def sort_simplex(self,func_vertices,vertices_mat):

        # For minimization problems: 
        # sort the objective function from small to large 
        # So, best_objective function is the minima of all objective function 

        if ( self.mode == "minimize"):  

            self.best_indx = 0 

            self.worst_indx = -1  

            self.lousy_indx = -2 
    
        elif ( self.mode == "maximize"): 

            self.best_indx = -1              
    
            self.lousy_indx = 1 

            self.worst_indx = 0  

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

            func_vertices[i] = self.f_obj.compute(in_parameters)
        
        return func_vertices   

    def Centroid(self):  
    
        # select all vertices except the worst vertex 

        except_worst = self.vertices_sorted[:self.worst_indx,:]  

        # compute the geometric center 

        return np.mean(except_worst,axis=0)
            
    def Reflect(self,centroid): 
        
        reflected_vetertex = centroid + self.alpha*(centroid - self.worst_vertex) 
       
        self.constrain(reflected_vetertex) 
 
        return reflected_vetertex  

    def Accept(self,vertex,func_vertex):  

        # subsitude worst vertex 

        self.vertices_sorted[self.worst_indx,:] = vertex  

        self.func_vertices_sorted[self.worst_indx] = func_vertex 

    def Expand(self,reflected,centroid):  

        expanded_vertex = centroid + self.kai*(reflected - centroid ) 

        self.constrain(expanded_vertex)

        return expanded_vertex 

    def Outside_Contract(self,centroid,reflected_vertex):  

        outside_vertex = centroid + self.gamma*(reflected_vertex - centroid) 

        self.constrain(outside_vertex) 

        return outside_vertex  
        
    def Inside_Contract(self,centroid):    

        inside_vertex = centroid + self.gamma*( self.worst_vertex - centroid) 

        self.constrain(inside_vertex) 

        return inside_vertex  
    
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

    def run_optimization(self):  

        # optimization procedures: 

        for itera in range(self.in_para.max_iteration):  
        
            # Ordering
            # Ascending ( minimization of objective )   
            # Descending ( maximization of objective ) 
          
            self.check_termination(itera)

            # Centroid 

            centroid = self.Centroid() 
            
            # Reflection 
            
            reflected_vertex = self.Reflect(centroid) 
            
            func_reflect = self.f_obj.compute(reflected_vertex) 
            
            if ( self.best <= func_reflect < self.lousy ): 

                self.Accept(reflected_vertex,func_reflect)   
                
                self.sort_simplex(self.func_vertices_sorted,self.vertices_sorted) 
                
                continue 
            
            # Expansion
            
            if ( func_reflect < self.best ): 

                expanded_vertex = self.Expand(reflected_vertex,centroid)          

                func_expand = self.f_obj.compute(expanded_vertex)

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

                    func_out_contract = self.f_obj.compute(outside_contract_vertex)

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
    
                    func_inside_contract = self.f_obj.compute(inside_contract_vertex)

                    if ( func_inside_contract < self.worst ):     
                        
                        self.Accept(inside_contract_vertex,func_inside_contract) 

                        self.sort_simplex(self.func_vertices_sorted,self.vertices_sorted) 
        
                        continue 
                
                    else:               
                        
                        self.Shrink()

                        continue    
            
                    
    def check_termination(self,n_itera): 

        self.optimze_logger.debug("Class NelderMeadSimplex:terminate function entered successfully !")

        if ( (np.amax(self.func_vertices_sorted)/np.amin(self.func_vertices_sorted) - 1)< self.in_para.obj_tol ):
        
            sci_obj = "{0:.1e}".format(self.in_para.obj_tol) 
            
            self.optimze_logger.info("Convergence criterion 1 is met: Ratio of obj_max/obj_min -1  < %.6f !\n"%sci_obj) 
        
            self.optimze_logger.info( "Optimization converges and program exits ! \n") 

            #optimization_output(best_para,self.func_vertices_sorted[0])             

            sys.exit()
    
        unique_obj,repeat = np.unique(self.func_vertices_sorted,return_counts=True) 

        if ( unique_obj.size < self.func_vertices_sorted.size ):

            self.optimze_logger.info("Convergence criterion 2 is met: some objective functions of different vertex begin to converge" )

            self.optimze_logger.info(" ".join(str(obj) for obj in vertices_sorted) )

            self.optimze_logger.info( "Optimization converges and program exits ! \n")

            #optimization_output(best_para,self.func_vertices_sorted[0])

            sys.exit()

        if ( np.all(np.std(self.func_vertices_sorted) < self.in_para.para_tol ) ):

            sci_para = "{0:.1e}".format(self.in_para.para_tol)

            self.optimze_logger.info("Convergence criterion 3 is met: the standard deviation of force-field paramteters across all vertices is %s  !\n"%sci_para)

            self.optimze_logger.info( "Optimization converges and program exits ! \n")

            #optimization_output(best_para,self.func_vertices_sorted[0])

            sys.exit()

        if ( n_itera  == self.in_para.max_iteration ): 

            self.optimze_logger.info("Convergence criterion 4 is met: Maximum number of iteration is reached !\n") 

            #optimization_output(best_para,self.func_vertices_sorted[0]) 

            sys.exit( "Maximum iteration %d is reached and Program exit !"%self.in_para.max_iteration)

        self.optimze_logger.debug("Class NelderMeadSimplex:terminate function exit successfully !")
            
        return None 

