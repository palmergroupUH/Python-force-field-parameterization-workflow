import logging 
import argparse 
import numpy as np 
import sys 
import IO.check_type  

class from_command_line():

    @classmethod
    def __init__(cls,jobID=None,total_cores=None,input_file=None,mode=None): 
    
        if ( mode is None ):  

            cls.MODE = "run"

        else: 

            cls.MODE = mode 
    
        if ( jobID is not None ):

            cls.JOBID = str(jobID) 
            
            cls.logger = cls.Set_Run_Mode(cls.JOBID + ".log",cls.MODE)
        
        if ( total_cores is not None ): 
            
            cls.TOTAL_CORES = total_cores
        
        if ( input_file is not None ): 

            cls.INPUT = input_file  
        
        all_options = np.array([ total_cores ,jobID ,input_file ])

        # if None of total_cores ,jobID ,input_file assigned, then use the command line options

        if ( np.all(all_options  == None ) ):  

            cls.Take_Command_Line_Args() 

            cls.set_global()
        
        if ( np.any(all_options  != None ) 
            and np.any(all_options == None) ):  

            sys.exit("ERROR: either assign all values as arguments or read all input from command line")    

        # check the following mandatory attributes  
        
        cls.check_necessary_attributes("JOBID")
        cls.check_necessary_attributes("TOTAL_CORES")
        cls.check_necessary_attributes("INPUT")
        cls.check_necessary_attributes("logger")
        
        # check the type of user-provided input: 

        cls.check_total_cores() 
        

        return None 

    @classmethod
    def finish_reading(cls): 

        return cls.logger,cls.TOTAL_CORES, cls.INPUT, cls.JOBID

    @classmethod
    def check_necessary_attributes(cls,attribute): 

        if ( not hasattr(cls,attribute) ):     

            sys.exit('global variable: "%s" not found in either command line or passed argument'%attribute)

        return None  

    @classmethod
    def check_total_cores(cls): 

        if ( not IO.check_type.is_int(cls.TOTAL_CORES)):  

            cls.logger.error("ERROR: varable: 'total_cores' must be an integer ! ")
        
            sys.exit("Check errors in log file ! ") 
    
        return None 

    @classmethod 
    def Take_Command_Line_Args(cls): 
        
        parser = argparse.ArgumentParser()

        parser.add_argument("-c", "--cores", type=int, required=True)

        parser.add_argument("-i", "--input", type=str, required=True)

        parser.add_argument("-j", "--job", type=str, 
                            required=True,help="Specify a unique jobid associated with each working folder and log file") 
        
        parser.add_argument("-m", "--mode", type=str, 
                            required=False,default="run",
                            help="choose run or debug mode")

        parser.add_argument("-Ref", "--ReferenceData", type=str, 
                            required=False,default="../ReferenceData",
                            help="provide the path of Reference data folder")

        parser.add_argument("-prep", "--prepsystem", type=str, 
                            required=False,default="../prepsystem",
                            help="provide the path of prepsystem folder")
        args = parser.parse_args()

        cls.argument = dict( args.__dict__.items() )  

        return None  

    @classmethod
    def set_global(cls):  

        cls.JOBID = cls.argument["job"]  
    
        cls.TOTAL_CORES = cls.argument["cores"] 

        cls.INPUT = cls.argument["input"] 

        cls.logger = cls.Set_Run_Mode(cls.JOBID +".log",cls.MODE) 

        return None  

    @classmethod
    def Select_Run_Mode(cls,arg): 

        mode = { 
        
        "debug": logging.DEBUG, 
        "run": logging.INFO

        }  
        
        return mode[arg] 

    @classmethod
    def Select_Formatter(cls,arg): 

        mode = { 
        
        "debug": "%(asctime)s - %(name)s - %(levelname)s - %(message)s", 
        "run": "%(message)s" 
        }  

        return mode[arg] 
        
    @classmethod
    def Set_Run_Mode(cls,logname,mode): 

        logger = logging.getLogger() 

        logger.setLevel(cls.Select_Run_Mode(mode)) 

        fh = logging.FileHandler(logname,mode="w")

        formatter = logging.Formatter(cls.Select_Formatter(mode)) 

        fh.setFormatter(formatter) 

        logger.addHandler(fh) 

        return logger 

