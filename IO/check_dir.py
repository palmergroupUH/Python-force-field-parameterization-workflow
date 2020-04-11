import os 
import sys 
import time 
import logging

def mkdirs_if_not_exist(folder_path,overwrite):  

    folder_create = logging.getLogger(__name__)

    if ( not os.path.isdir(folder_path)): 

        os.makedirs(folder_path) 

        return None  
   
    if ( overwrite ):  

        os.system("rm -r %s"%folder_path)  
    
        time.sleep(0.5) 
    
        os.makedirs(folder_path) 
   
    elif ( not overwrite and os.path.isdir(folder_path)):   

        folder_create.error("The folder: '%s' already exists ! It can not be overwritten; please set 'overwrite=True' "%folder_path)
        
        sys.exit("check errors in log file")

    return None 
    
def check_folder_status(folder_path):  

    exist = True 

    empty = False 

    if ( not os.path.isdir(folder_path)): 

        exist = False

        return exist,empty

    content = os.listdir(folder_path) 

    if ( len(content ) == 0 ):

        empty = True 

    return exist,empty 

def decide_folder_status(folder_path):

    exist,empty = check_folder_status(folder_path)

    folder_status = logging.getLogger(__name__)  

    if ( not exist ): 
         
        folder_status.error("ERROR: Folder: %s does not exist "%folder_path)

        sys.exit("Check errors in log file !") 

    if ( empty ): 

        folder_status.error("ERROR: Folder: %s  is empty !  "%folder_path)

        sys.exit("Check errors in log file !") 
       
    return None  
        
        
        
        

