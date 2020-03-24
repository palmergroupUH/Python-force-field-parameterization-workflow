import numpy as np 
import os 
import sys 

class parse_input():
    
    def __init__(self,input_file):

        if ( not os.path.isfile(input_file) or os.path.getsize(input_file) ==0  ):

            sys.exit("Make sure the input file exists and not empty")

        else:

            self.input_file = input_file 

        return None 

    def is_int(self,a):

        try:

            int(a)

        except ValueError:

            return False

        return True

    def is_float(self,a):

        try:

            float(a)

        except ValueError:

            return False

        return True

    def is_string(self,a):

        if ( not self.is_float(a) and not self.is_int(a) ):
    
            return True

        else:

            return False

    def contain_no_number(self,contents):

        not_number = 0 

        for i in range(len(contents)):
        
            if ( not self.is_float(contents[i]) and not self.is_int(contents[i]) ):

                not_number +=1  

        if ( not_number == len(contents)):
    
            return True 

        else:
        
            return False

    def its_units(self,contents):

        item = 0 

        if ( self.contain_no_number(contents)):

            item += 1 

        if ( len(contents) == 1  ):

            item +=1  

        if ( item ==2 ): 

            return True

        else: 

            return False

    def its_objective(self,contents):

        try: 

            results = np.array([ self.is_string(contents[0]), 
                                 self.is_string(contents[1]), 
                                 self.is_float(contents[2]),
                                 self.is_int(contents[3]),
                                 self.is_int(contents[4])] ) 

            if ( np.all(results) ): 

                return True 
    
            else:

                return False

        except ( IndexError): 

            return False
            
    def its_command(self,contents):

        if ( self.contain_no_number(contents)):    

            return True 

    def identify(self,contents):

        if ( self.its_units(contents)): 

            return "objective"

        elif ( self.its_objective(contents)):
        
            return "objective" 

        elif ( self.its_command(contents)):

            return "command"

        else:

            return "optimizer"

        return None 

    def read(self,skip_lines=None,stop_after=None): 

        if ( skip_lines is None ): 

            skip_lines = 0  

        with open(self.input_file,"r") as content:

            for line in range(skip_lines): 
                
                content.readline() 

            line_number = 1 

            input_files = {}

            stop_counter = 0 
            
            for line in content.read().splitlines():
                
                contents = line.split() 

                stop_counter += 1  

                if ( stop_after is not None and stop_after == stop_counter ): 

                    break

                # skip the empty lines 

                if ( contents == [] ): 

                    line_number += 1 
        
                    continue  

                elif ( "#" in contents[0] or "&" in contents[0]): 

                    line_number += 1 

                    continue 

                else:

                    input_files[line_number] = contents 

                    line_number += 1

            return input_files 

parse = parse_input("in_optimize")          

print (parse.read())


