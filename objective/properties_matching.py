import numpy as np 
import os 
import sys 
import logging 
import IO.input_file
import  

class setup(): 

    def __init__(self,input_file,
                     simulation_engine,
                     skip_lines=None,
                     stop_after=None): 

        self.objective_logger = logging.getLogger(__name__) 
   
        self.input_data_dict = IO.input_file.parse(input_file,skip_lines,stop_after)  

        self.parse_units()

	def This_is_matching_type():  

		results = [ is_string(a[0]), is_string(a[1]), is_float(a[2]) , is_int(a[3]), is_int(a[4]) ] 

		return None 

	def objective_function_format(argument): 

		this_line_is_matching(argument) 	
		

		return None 

    def folder_setup():  

            

    def parse_units(self): 

        self.units = self.input_data_dict[0] 

        return None 

    def parse_objective_function(self):      

        return None 
       
    def check_reference_data(self): 
        
        
        return None 

    def check_prep_system(self): 

        return None 


