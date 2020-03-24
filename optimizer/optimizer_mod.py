import numpy as np 
import logging
import sys 
import IO.input_file
import IO.type_check


#class parse_input(): 

class set_optimizer: 
    
    def __init__(self,input_file,logname,skipped=None,stop_after=None): 

        # create logger for the optimization jobs

        self.optimizer_logger = logging.getLogger(__name__)   

        self.add_logger(logname) 
        
        # parse the input file

        self.input_data_dict = IO.input_file.parse(input_file,skipped,stop_after)

        self.keys_lst = list(self.input_data_dict.keys())
        
        self.values_lst = list(self.input_data_dict.values() ) 

        # following the order in input file parse all input values

        self.parse_guess()

        self.parse_fit_and_fixed()

        self.parse_constraints()

        self.parse_termination()

        self.parse_optimizer()  

        self.check_input_parameters()  

    def add_logger(self,logname): 

        if ( len(self.optimizer_logger.handlers) == 0 ): 

            self.optimizer_logger.setLevel(logging.INFO)              
    
            fh = logging.FileHandler(logname,mode="w") 

            formatter = logging.Formatter("%(message)s") 
            
            fh.setFormatter(formatter) 

            self.optimizer_logger.addHandler(fh)

        return None 

    def parse_dumping_freq(self):

        dump_freq = self.input_data_dict[self.keys_lst[0]] 

        try: 
        
            self.dump_para = np.array(dump_freq).astype(np.int )   

        except ( ValueError, TypeError ): 
        
            self.optimzer_logger.info("type and value errors in reading dump frequency" ) 
    
            sys.exit("Check errors in log file !") 

        return None 

    def parse_guess(self):

        guess_parameters = self.input_data_dict[self.keys_lst[0]] 
    
        all_parameters = [] 

        for guess in guess_parameters:
        
            if ( IO.type_check.is_string(guess) ):

                continue  
    
            else:

                all_parameters.append(guess)  
    
        # convert all guess parameters to numpy float 64

        self.guess_parameter = np.array(all_parameters).astype(np.float64) 

    def parse_fit_and_fixed(self):

        fit_and_fix = self.input_data_dict[self.keys_lst[1]]

        try:  

            self.fit_and_fix = np.array(fit_and_fix).astype(np.float64) 

        except (ValueError, TypeError ):

            self.optimzer_logger.info("type and value errors in fit_and_fixed variables")  
             
            sys.exit("Check errors in log file !") 

        self.fit_index = np.array([ i for i,x in enumerate(self.fit_and_fix) if x == 1  ],dtype=np.int) 

        self.unfit_index = np.array([ i for i,x in enumerate(self.fit_and_fix) if x ==0 ],dtype=np.int) 

        self.check_guess_parameters()  

    def parse_constraints(self):  

        self.constraints = self.input_data_dict[self.keys_lst[2]]  

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

                    self.constraints_fit_index[nindex] -= num_shift 

                self.constraints_bound = np.array([ [ self.constraints[3*indx+1], self.constraints[3*indx+2]] for indx in range(num_constraints)]) 

            except ( ValueError, TypeError) :          

                self.optimizer_logger.error("ERROR: Type or Value errors in constraints parameters")

                sys.exit("Check errors in log file !")

            self.check_constraints()

            self.constraints_bound.astype(np.float64)

        return None 

    def parse_termination(self):

        optimize_settings = self.input_data_dict[self.keys_lst[3]]
        
        try: 
        
            self.max_iteration = int(optimize_settings[0]) 

            self.para_tol = float(optimize_settings[1]) 

            self.obj_tol = float(optimize_settings[2])
       
        except ( ValueError, TypeError): 
 
            self.optimizer_logger.error("ERROR: Termination conditon format should be:"
                                       " integer, float,float")
        
            sys.exit("Check errors in log file !")

        return None 

    def parse_optimizer(self):

        optimizer_settings = self.input_data_dict[self.keys_lst[4]] 

        num_arguments = len(self.keys_lst) 

        try: 
       
            # specific optimizer used: Nelder-Mead simplex, Levenberg–Marquardt, Gauss-Newton ... 

            self.optimizer_type =  optimizer_settings[0] 

            # oher possible arguments:  

            self.optimizer_argument = optimizer_settings[0:] 

            # rest of arguments ofr the optimizer: 

            self.optimizer_input = self.values_lst[5:num_arguments]

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

                    self.optimze_logger.info( "Lower constraints are applied...")
                    self.optimze_logger.info( "Parameter: "     
                                              + str( array[self.constraints_fit_index[i]]) 
                                              + "  is constrained to " + str( lower))

                    array[self.constraints_fit_index[i]] = lower
        
                # evaluate the expression: lower bound < para

                if ( eval(constraints_upper_expr)):

                    # lower bound is indeed < para

                    pass

                else:

                    self.optimze_logger.info( "Upper constraints are applied..." )

                    self.optimze_logger.info( "Parameter: " 
                                              + str( array[self.constraints_fit_index[i]]) 
                                              + "  is constrained to " + str( upper))

                    array[self.constraints_fit_index[i]] = upper

        return None

    # combined fitted and fixed parameters into full parameters ( same length as guess parameters )  

    def regroup_with_fixed(self,fitted_para): 
    
        para_all= np.zeros(self.guess_parameter.size)

        para_all[self.fit_and_fix==0] = self.guess_parameter[self.fit_and_fix==0] 
        
        para_all[self.fit_and_fix==1] = fitted_para  

        return para_all  

if (__name__ == "__main__"): 

    test = optimizer("in_para",skipped=16)

    print (test.parse_guess())

    print ( test.parse_optimizer())
        
    print ( test.optimizer_input) 
