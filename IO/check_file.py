import os 

def status_is_ok(filename): 

    # file does not exist ! 

    if ( not os.path.isfile(filename)): 
        
        return False 

    # file is empty !  

    if ( os.stat(filename).st_size ==0 ): 

        return False	

    # status is ok 

    return True 

def file_size_is_too_big(filename,threshold_size,units): 

    file_size_bytes = os.stat(filename).st_size 

    if ( units == "MB"): 

        # bit shift ( convert bytes into MB )  

        file_size_MB = file_size_bytes >> 20  

        if ( file_size_MB > threshold_size ):  

            return True 

    elif ( units =="GB"): 

        file_size_GB = file_size_bytes >> 30         

        if ( file_size_GB > threshold_size ):

            return True  

    return False 
		
 
