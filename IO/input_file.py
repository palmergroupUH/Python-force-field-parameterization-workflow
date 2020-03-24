import os 
import sys 

def parse(filename,skip_lines=None,stop_after=None): 

    if ( not os.path.isfile(filename) ): 
        
        sys.exit("input file does not exist") 

    if ( skip_lines is None ): 

        skip_lines = 0  

    with open(filename,"r") as content:

        for line in range(skip_lines): 
            
            content.readline() 

        line_number = 1 

        input_files = {}

        stop_counter = 0 
        
        for line in content.read().splitlines():
            
            contents = line.split() 

            stop_counter += 1  

            # stop reading the file at certain lines 

            if ( stop_after is not None and stop_after == stop_counter ): 

                break

            # skip the empty lines 

            if ( contents == [] ): 

                line_number += 1 
    
                continue  

            # skip the line starting with "# or & " which are comments
 
            elif ( "#" in contents[0] or "&" in contents[0]): 

                line_number += 1 

                continue 

            # reading the content: 

            else:

                input_files[line_number] = contents 

                line_number += 1

        return input_files 
