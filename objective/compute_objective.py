# Python standard library: 
import numpy as np
import os 
import logging 
import sys
import importlib
# Local library: 
import IO
import objective
# Third-party libraries: 

class prepare(): 

    def __init__(self,JOBID,ref_dict,predict_dict,arg_dict,sampling_method): 

        self.output_folder = os.path.join(JOBID,"Output")

        self.logger = logging.getLogger(__name__) 

        self.sampling = sampling_method 

        self.check_sampling_method() 

        self.load_each_matching_module(ref_dict,predict_dict,arg_dict ) 

        return None  

    def check_sampling_method(self):

        # check if the sampling method has attributes:
        # 
        if ( not hasattr(self.sampling,"run")): 

            self.logger.error("The sampling_method object does not have 'run' attributes")

            sys.exit("Check errors in the log file")
        
        if ( not hasattr(self.sampling,"exit")):

            self.logger.error("The sampling_method object does not have 'exit' attributes")

            sys.exit("Check errors in the log file")

        return None 

    def input_keyword_to_module_keyword(self,input_keyword):

        keyword_dict = { 
                        "force": "force_matching",
                        "rdf": "rdf_matching",
                        "isobar": "isobar_matching"

                        } 

        return keyword_dict[input_keyword]  

    def load_each_matching_module(self,ref_dict,predict_dict,arg_dict): 
    
        self.load_objective_lst = [ ] 

        for every_type,_,_ in zip(ref_dict.keys(),predict_dict.keys(),arg_dict.keys()): 

            module_name = self.input_keyword_to_module_keyword(every_type)
    
            import_path = "objective"+"."+"%s"%module_name+"."+"%s"%module_name

            loaded_matching = importlib.import_module(import_path)
    
            self.check_loaded_module(loaded_matching)

            ref_sub_dict = ref_dict[every_type] 
    
            predict_sub_dict = predict_dict[every_type] 

            arg_sub_dict = arg_dict[every_type]

            for keys,_,_ in zip(ref_sub_dict.keys(),predict_sub_dict.keys(),arg_sub_dict.keys()): 

                ref_address = ref_sub_dict[keys] 

                predict_address = predict_sub_dict[keys] 

                argument = arg_sub_dict[keys]

            initialize_objective = loaded_matching.load(ref_address,predict_address,argument)
            
            self.check_objective_attribute(initialize_objective)

            self.load_objective_lst.append( initialize_objective )

        return None 

    def check_loaded_module(self,loaded_module): 

        if ( not hasattr(loaded_module,"load")): 

            self.logger.error("The loaded matching module does not have the 'load' attributes")

            sys.exit("Check errors in the log file")

        return None 

    def check_objective_attribute(self,objective):

        if ( not hasattr(objective,"optimize") ):

            self.logger.error("The loaded objective function does not have the 'optimize' attributes")

            sys.exit("Check errors in the log file")

        return None 

    def optimize(self,fftype,force_field_parameters,status): 

        # run sampling: 

        self.sampling.run(fftype,force_field_parameters) 

        # check sampling status: 

        job_successful = self.sampling.exit() 

        if ( not job_successful ): 

            self.logger.error("Sampling method does not exit successfully !")

            sys.exit("Check errors in the log file")

        sum_objective = 0 

        for job in self.load_objective_lst:  

            sum_objective += job.optimize() 

            # if the rename file attributes exist, rename the file needed

            if ( hasattr(job,'rename')): 

                job.rename(status,self.output_folder) 

        return sum_objective  

    def update(self,current_obj,best_obj,status):

        if (current_obj < best_obj):  

            for job in self.load_objective_lst: 

                if (hasattr(job,'update')): 

                    job.update(status,self.output_folder) 

        return None 



    
