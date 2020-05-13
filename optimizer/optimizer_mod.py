# Python standard library: 
import numpy as np 
import logging
import sys 
import os 

# Local library: 
import IO.input_file
import IO.check_type

# Third-party libraries: 

#class parse_input(): 

class set_optimizer: 
       
    """Description: This is a parent class that will be inherited by 
    all other optimizer classes (gradient-free or gradient) to do the followings: 

     1. parse a input file with a predetermined format, 
     2. method to constrain certain fitted parameters within bounds  
     3. method to recombine fixed and fitted parameters before 
        passing them to objective function    
     4. method to write optimizer output 


     Inherited variables descriptions:

     self.dump_para ( arraym integer ):  
        an array of integer that specify how frequent to dump the restart file 
        and best parameters.  

     self.optimizer_logger (object): 
         an logger object that dump all output into a log file. 

     self.para_type_lst ( str list ):
        a string defining the type of guess parameters    

     self.guess_parameters (array,np.float64): 
         an array of guessed floating point value given by users  

     self.fit_and_fix ( array, integer ): 
         an array with value either 1 or 0 to determine which parameters to be fixed or fitted 

     self.constraints_index ( array, integer ): 
         an array of index value determine which guess parameters 
         to be constrained:  

         e.g. 1 means 1st guess parameters, 10 means 10th guess parameters 
    
     self.constraints_bound ( array, np.float64): 
         an array of values size 2 with lower and upper 
   
     self.constraints_fit_index ( array, integer ):
         an array of index value determine which guess parameters
         to be constrained ( index may be adjusted depending on which  
         arameters are fitted and which are fixed ): 

    self.optimizer_type ( a string ): 
        The optimizer name: Nelder-Mead simplex, Levenberg–Marquardt ...  
       
    self.optimizer_argument ( a string ):   
        The optimizer argument associated with spcecific optimizer   

    self.optimizer_input ( a input argument ):  
        The optimizer input: input values used by optimizers ( optional )     

    Inherited method descriptions: 
    
    self.regroup_with_fixed():  
        recombine the fitted parameters with the fixed parameters into
        an array of full parameters.   
        
    self.constrain(): 
        perform constrain operations on some fitted parameters  

    self.write_optimizer_output():  

    self.print_optimizer_log():  
    
    self.optimizer_restart_content(): 
    """
 
    def __init__(self,input_file,logname=None,skipped=None,stop_after=None): 

        # create logger for the optimization jobs

        self.add_logger(logname) 
        
        # parse the input file
    
        # since restart file requires all the content, read the content before skipped 
        self.input_data_non_optimizer_dict = IO.input_file.parse(input_file,0,skipped,include_comment=True)
        
        # read the conetnt for optimizer
        self.input_data_dict = IO.input_file.parse(input_file,skipped,stop_after)

        self.keys_lst = list(self.input_data_dict.keys())
        
        self.values_lst = list(self.input_data_dict.values() ) 

        # following the order in input file parse all input values
        # self.pointer: point to current line in the input file  
        # self.pointer += 1 for each line read 
        # set pointer to the begining 

        self.pointer = 0 

        self.parse_dumping_freq()     

        self.parse_guess()

        self.parse_fit_and_fixed()

        self.parse_constraints()

        self.parse_termination()

        self.parse_optimizer() 

        self.check_input_parameters() 

    def add_logger(self,logname): 
        
        if ( logname is not None ):   
            
            self.optimizer_logger = logging.getLogger(__name__) 

            self.optimizer_logger.setLevel(logging.INFO)              
    
            fh = logging.FileHandler(logname,mode="w") 

            formatter = logging.Formatter("%(message)s") 
            
            fh.setFormatter(formatter) 

            self.optimizer_logger.addHandler(fh)

        else: 
            
            self.optimizer_logger = logging.getLogger(__name__) 

        return None 

    # dumping the current best parameters, restart simplex, 

    def write_optimizer_output(self,freq,n_iteration,output_address,filename,mode,content_dict):  

        if ( n_iteration%freq == 0 ):  

            if ( output_address is None ): 

                outputfile = filename 

            elif ( os.path.isdir(output_address) ):  

                #os.makedirs(output_address)

                outputfile = os.path.join(output_address,filename) 

            self.write_output_to_file(n_iteration,outputfile,mode,content_dict)

        return None 

    # write output:  
    
    def write_output_to_file(self,n_iteration,outputfile,mode,content):  

        if ( mode =="a" and n_iteration ==0 ): 
            
            open(outputfile,"w").close() 

        with open(outputfile,mode) as output:  

            for line in content.keys(): 
                
                output.write( content[line] )  

        return None 

    def parse_dumping_freq(self):

        dump_freq = self.input_data_dict[self.keys_lst[self.pointer]] 

        try: 
        
            self.dump_para = np.array(dump_freq).astype(np.int )   

        except ( ValueError, TypeError ): 
        
            self.optimizer_logger.info("type and value errors in reading dump frequency" ) 
    
            sys.exit("Check errors in log file !") 
    
        # update pointer: only 1 line is read and so point to next line  

        self.set_dump_freq() 

        self.pointer += 1 

        return None 

    def set_dump_freq(self): 

        if ( self.dump_para.size < 2 ):

            self.optimizer_logger.error("ERROR: At least two dump frequency" 
                                        "arguments are needed: "
                                        "One for restart,"
                                        "One for optimization parameters ")
                       
            self.output_freq = self.dump_para[0] 

            self.restart_freq = self.dump_para[0] 
                
        else: 

            self.output_freq = self.dump_para[0] 
            
            self.restart_freq = self.dump_para[1]

        return None 

    def parse_guess(self):

        guess_parameters = self.input_data_dict[self.keys_lst[self.pointer]] 
    
        all_parameters = [] 
    
        self.para_type_lst = [] 

        for guess in guess_parameters:
        
            if ( IO.check_type.is_string(guess) ):

                self.para_type_lst.append(guess)  

                continue  
    
            else:

                all_parameters.append(guess)  
    
        # convert all guess parameters to numpy float 64

        self.guess_parameter = np.array(all_parameters).astype(np.float64) 

        # update pointer: only 1 line is read and so point to next line  

        self.pointer += 1 
        
        return None 

    def parse_fit_and_fixed(self):

        fit_and_fix = self.input_data_dict[self.keys_lst[self.pointer]]

        try:  

            self.fit_and_fix = np.array(fit_and_fix).astype(np.float64) 

        except (ValueError, TypeError ):

            self.optimizer_logger.info("type and value errors in fit_and_fixed variables")  
             
            sys.exit("Check errors in log file !") 

        self.fit_index = np.array([ i for i,x in enumerate(self.fit_and_fix) if x == 1  ],dtype=np.int) 

        self.unfit_index = np.array([ i for i,x in enumerate(self.fit_and_fix) if x ==0 ],dtype=np.int) 

        self.check_guess_parameters()  

        # update pointer: only 1 line is read and so point to next line  

        self.pointer += 1 
       
        return None  

    def parse_constraints(self):  

        self.constraints = self.input_data_dict[self.keys_lst[self.pointer]]  

        num_constraints = int(len(self.constraints)/3) 

        if ( self.constraints[0] == "None" or self.constraints[0] == "none"): 
             
            self.constraints_index = np.array([]) 

            self.constraints_fit_index = np.array([]) 

            self.constraints_bound = np.array([])  
       
        else:

            try: 
                self.constraints_index = np.array([self.constraints[idx*3]  
                                        for idx in range(num_constraints)]).astype(np.int)-1  
                
                self.constraints_fit_index = np.zeros(self.constraints_index.size).astype(np.int) 

                for nindex in range(self.constraints_index.size): 

                    num_shift = sum( i < self.constraints_index[nindex] for i in self.unfit_index) 

                    self.constraints_fit_index[nindex] = self.constraints_index[nindex] - num_shift 

                    
                self.constraints_bound = np.array([ [ self.constraints[3*indx+1], self.constraints[3*indx+2]] for indx in range(num_constraints)]) 

            except ( ValueError, TypeError) :          

                self.optimizer_logger.error("ERROR: Type or Value errors in constraints parameters")

                sys.exit("Check errors in log file !")

            self.check_constraints()

            self.constraints_bound.astype(np.float64)

        # update pointer: only 1 line is read and so point to next line  

        self.pointer += 1 

        return None 

    def parse_termination(self):

        optimize_settings = self.input_data_dict[self.keys_lst[self.pointer]]
        
        try: 
        
            self.max_iteration = int(optimize_settings[0]) 

            self.para_tol = float(optimize_settings[1]) 

            self.obj_tol = float(optimize_settings[2])
       
        except ( ValueError, TypeError): 
 
            self.optimizer_logger.error("ERROR: Termination conditon format should be:"
                                       " integer, float,float")
        
            sys.exit("Check errors in log file !")

        # update pointer: only 1 line is read and so point to next line  

        self.pointer += 1

        return None 

    def parse_optimizer(self):

        optimizer_settings = self.input_data_dict[self.keys_lst[self.pointer]] 

        try: 
       
            # specific optimizer used: Nelder-Mead simplex, Levenberg–Marquardt, Gauss-Newton ... 

            self.optimizer_type =  optimizer_settings[0] 

            # oher possible arguments:  

            self.optimizer_argument = optimizer_settings[0:] 

            # rest of arguments ofr the optimizer: 

            self.optimizer_input = self.values_lst[self.pointer+1:]

        except( ValueError, TypeError):

            self.optimizer_logger.error("ERROR: optimizer needs at least one argument !") 

            sys.exit("Check errors in log file !")

    def check_input_parameters(self): 

        # guess parse input parameters 

        # The followings are Mandatory optimization parameters passed to any optimizer !!     

        self.attribute_exist("guess_parameter")  

        self.attribute_exist("fit_and_fix")  

        self.attribute_exist("constraints_fit_index")
            
        self.attribute_exist("constraints_bound") 

        self.attribute_exist("max_iteration") 

        self.attribute_exist("para_tol")

        self.attribute_exist("obj_tol")

        return None 

    def attribute_exist(self,attribute): 

        if ( not hasattr(self,attribute) ):

            self.optimizer_logger.error('No attribute "%s" found in parsed input object'%attribute) 
    
            sys.exit("Check errors in log file")
    
        return None  

    def check_constraints(self):

        # at least 3 arguments needed if any guess parameters to be constrained 

        if  ( len(self.constraints)%3 != 0 or len(self.constraints) < 3 ): 
           
            self.optimizer_logger.error( "ERROR: At lesat 3 arguments needed to constrain a guess parameter")  

            sys.exit("Check errors in log file")

        # constraint indexes can not be more than the number of guess parameters  

        for i in range(self.constraints_index.size):  

            if ( np.amax(self.constraints_index) > self.guess_parameter.size -1 ):   
           
                self.optimizer_logger.error( "ERROR: The value of constraint index" 
                                        "should be < number of guess of parameters")  

                sys.exit("Check errors in log file")

        for cindex in self.constraints_index:  

            if ( cindex  in self.unfit_index): 

                self.optimizer_logger.error( "ERROR:"
                                             "constraint index has to be: " 
                                             "fitted variable (=1)." 
                                             "Fixed variable (=0) " 
                                             "can not be constrained") 

                sys.exit("Check errors in log file")
        
        return None 

    def check_guess_parameters(self ): 

        # The number of guess parameter should be equal to fitted + fixed  

        if ( self.guess_parameter.size != self.fit_and_fix.size ): 

            self.optimizer_logger.error("ERROR: The number of guess parameters is not" 
                                    "equal to the nubmer of fitted parameters (=1)" 
                                    "+ the number of fixed parameters (=0) " )

            sys.exit("Check errors in log file")

        # either fit (1) or fixed (0) ; No other number is allowed  

        if ( np.any(self.fit_and_fix > 1 ) or np.any(self.fit_and_fix < 0 )) : 

            self.optimizer_logger.error( "ERROR: The fit or fixed parameter" 
                                            "should only be either 1 or 0" )
   
            sys.exit("Check errors in log file")  

        return None 
    
    def constrain(self,array): 
        
        num_constraints = self.constraints_fit_index.size 

        num_criterion = np.size(self.constraints_bound,0) 
        
        if ( num_constraints == num_criterion and num_constraints > 0 ): 

            for i in range(num_criterion): 

                lower = self.constraints_bound[i][0] 
            
                upper = self.constraints_bound[i][1]

                constraints_lower_expr  = lower + "<=" + str(array[self.constraints_fit_index[i]])

                constraints_upper_expr  =  str(array[self.constraints_fit_index[i]]) + "<=" + upper

                # evaluate the expression: lower bound < para

                if ( eval(constraints_lower_expr)):

                    # lower bound is indeed < para 

                    pass

                else:

                    self.optimizer_logger.info( "Lower constraints are applied...")
                    self.optimizer_logger.info( "Parameter: "     
                                              + str( array[self.constraints_fit_index[i]]) 
                                              + "  is constrained to " + str( lower))

                    array[self.constraints_fit_index[i]] = lower
        
                # evaluate the expression: lower bound < para

                if ( eval(constraints_upper_expr)):

                    # lower bound is indeed < para

                    pass

                else:

                    self.optimizer_logger.info( "Upper constraints are applied..." )

                    self.optimizer_logger.info( "Parameter: " 
                                              + str( array[self.constraints_fit_index[i]]) 
                                              + "  is constrained to " + str( upper))

                    array[self.constraints_fit_index[i]] = upper

        return None

    # combined fitted and fixed parameters into full parameters ( same length as guess parameters )  

    def regroup_with_fixed(self,fitted_para): 
    
        para_all= np.zeros(self.guess_parameter.size,dtype=np.float64)

        para_all[self.fit_and_fix==0] = self.guess_parameter[self.fit_and_fix==0] 
        
        para_all[self.fit_and_fix==1] = fitted_para  

        return para_all  

    def print_optimizer_log(self):  

        num_fit = self.guess_parameter[self.fit_and_fix==1].size 
        
        num_fix = self.guess_parameter[self.fit_and_fix==0].size

        self.optimizer_logger.info("\n") 
        self.optimizer_logger.info("------------------------- Initialize Optimization Input Parameters -----------------------------------\n") 
        self.optimizer_logger.info("Optimizer: %s \n"%self.optimizer_type )
        self.optimizer_logger.info("-------------------------------------------------------------------------------------------------------\n")  
        self.optimizer_logger.info("Number of Vertices: %d \n"%(num_fit + 1 ))  
        self.optimizer_logger.info("-------------------------------------------------------------------------------------------------------\n")  
        self.optimizer_logger.info("Guess parameters are: \n " )  
        self.optimizer_logger.info(" ".join(str(para) for para in self.guess_parameter) + "\n" )  
        self.optimizer_logger.info("-------------------------------------------------------------------------------------------------------\n")  
        self.optimizer_logger.info("%d Fitting parameters are:  \n"  %(num_fit)) 
        self.optimizer_logger.info(" ".join(str(para) for para in self.guess_parameter[self.fit_index] ) + "\n")
        self.optimizer_logger.info("-------------------------------------------------------------------------------------------------------\n")
        self.optimizer_logger.info("%d Fixed parameters are: \n"%(num_fix) ) 
        self.optimizer_logger.info(" ".join(str(para) for para in self.guess_parameter[self.unfit_index]) + "\n" )
        self.optimizer_logger.info("-------------------------------------------------------------------------------------------------------\n")  
        self.optimizer_logger.info("%d constrained parameters : \n" %(self.constraints_index.size)) 

        for i in range(self.constraints_index.size): 

            self.optimizer_logger.info("The guess parameter: %.6f is constrained between %.6f and %.6f "%( float(self.guess_parameter[self.constraints_index[i]]),float(self.constraints_bound[i][0]),float(self.constraints_bound[i][1]) ) + "\n")

        self.optimizer_logger.info("-------------------------------------------------------------------------------------------------------\n")

        self.optimizer_logger.info(" ".join( str(i) for i in self.optimizer_argument ) + "\n")

        return None 

    def non_optimizer_restart_content(self,itera): 

        content = {} 

        content[0] = "# Current iteration %d: This is a restart file \n\n"%itera 

        for i,line in enumerate(self.input_data_non_optimizer_dict.keys()):   

            content[i+1] = " ".join(para for para in self.input_data_non_optimizer_dict[line]) + "\n" 
           
        return content,i  

    def optimizer_restart_content(self,itera,best_para):  

        # content of restart file following the same as the input file
        # dictionary: keys: line number, values: content of restart file 

        # write only general options:  
        content,total_lines = self.non_optimizer_restart_content(itera) 
        content[total_lines+1] = "# output and restart frequency \n\n"
        content[total_lines+2] = " ".join(str(para) for para in self.dump_para) + "\n\n" 
        content[total_lines+3] = "# This is the guess parameters: \n\n"
        content[total_lines+4] = "# " + " ".join(self.para_type_lst) + " ".join( str(ele) for ele in self.guess_parameter) + "\n\n" 
        content[total_lines+5] = "# This is the current best parameters: \n\n" 
        content[total_lines+6] = " ".join(self.para_type_lst) + " " + " ".join( str(para) for para in best_para) + "\n\n" 
        content[total_lines+7] = "# fit (1) and fix (0) parameters: \n\n" 
        content[total_lines+8] = " ".join(str(para) for para in self.fit_and_fix) + "\n\n"  
        content[total_lines+9] = "# constraints ( index lower-bound upper-bound ... ):\n\n "
        content[total_lines+10] = " ".join(str(para) for para in self.constraints) + "\n\n" 
        content[total_lines+11] = "#set termination criterion: max number of iteration, tolerance for parameters,tolerance for objective \n\n" 
        content[total_lines+12] = "%d "%self.max_iteration + " " +  "{0:.1e}".format(self.obj_tol) + " " + "{0:.1e}".format(self.para_tol) + "\n\n" 

        content[total_lines+13] = "# create (Perturb) or use existing vertices (Restart): \n\n"
        content[total_lines+14] = "# " + self.optimizer_type + " " + " ".join( para for para in self.optimizer_argument)+ "\n\n"
        content[total_lines+15] = "# " + " ".join(str(para) for para in self.optimizer_input) + "\n\n" 
        content[total_lines+16] = self.optimizer_type + " Restart \n\n"

        return content  

    def dump_restart(self,itera,filename_lst,output_address,restart_content_dict) :
        
        self.write_optimizer_output(self.restart_freq,
                                   itera,
                                   output_address,
                                   filename_lst[0],
                                   "a",
                                   restart_content_dict )

    
        self.write_optimizer_output(self.restart_freq,
                                   itera,
                                   output_address,
                                   filename_lst[1],
                                   "w",
                                   restart_content_dict )

        return None 

    def dump_best_objective(self,itera,filename,output_address,best_value): 

        best_objective = {} 
    
        best_objective[0] = str(best_value) + "\n"
        
        self.write_optimizer_output(self.output_freq, 
                                    itera,
                                    output_address,
                                    filename,
                                    "a", 
                                    best_objective)  

        return None 

    def dump_best_parameters(self,itera,filename,output_address,best_parameters):   

        best_para = {} 
       
        best_para[0] = " ".join(str(para) for para in best_parameters) 

        self.write_optimizer_output(self.output_freq, 
                                    itera,
                                    output_address,
                                    filename,
                                    "w", 
                                    best_para)  
        return None 
    
