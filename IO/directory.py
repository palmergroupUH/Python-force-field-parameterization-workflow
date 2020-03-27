import os 
import sys 
import subprocess 


class mkdirs( ): 

    ref_folder = "../ReferenceData"

    prep_folder = "../prepsystem"

    predict_folder = "Predicted"

    setup_logger = logging.getLogger() 

    def __init__(self,jobid,matching_in,overwrite=None): 

        self.setup_logger.debug("Start setting up the working folder ... ")

        if ( overwrite == True ):
        
            self.overwrite = True   

        else: 

            self.overwrite = False

        self.home = os.getcwd() 

        self.matching_type = np.unique([ everytype.split()[0]\
                             for everytype in matching_in ])    

        self.subfolders_per_type = self.Get_subfolders_from_type(self.matching_type,matching_in )       

        self.jobid = jobid 

        self.Check_Folders(self.ref_folder,self.matching_type,self.subfolders_per_type)

        self.Check_Folders(self.prep_folder,self.matching_type,self.subfolders_per_type)

        self.Mkdir() 

        self.setup_logger.debug("working folder is successfully set up ... ")

        return None 

    def Get_subfolders_from_type(self,matching_type,matching_in ): 

        subfolders_per_type = {} 
    
        for typename in matching_type:  

            type_sub = [] 
    
            for everytype in matching_in: 

                args = everytype.split() 

                if ( args[0] == typename): 

                    type_sub.append(args[1]) 
                
            subfolders_per_type[typename] = type_sub    
        
        return subfolders_per_type 

    def Check_Folders(self,address,matching_type,subfolders_per_type):

        folder_exists = os.path.isdir(address) 

        if ( not folder_exists ): 

            self.setup_logger.error( "ERROR: folder " + address + " does not exist") 

            sys.exit()

        if ( len(os.listdir(address+"/" )) == 0 )  : 

            self.setup_logger.error( "ERROR: No folders inside " + address  ) 
        
            sys.exit() 

        for match in matching_type: 

            for subfolder in subfolders_per_type[match]: 

                match_folder = address + "/" + match + "/" + subfolder 
                
                if ( not os.path.isdir( match_folder)): 

                    self.setup_logger.error( "ERROR: folders: " + match_folder+"\n" )
                    self.setup_logger.error( "Check the following: \n\n" )
                    self.setup_logger.error( "1. Does the type %s exist ?\n\n"%match )  
                    self.setup_logger.error( "2. Is this folder name consistent with the matching type name used in input file e.g. isobar, force etc ?\n\n")
                    self.setup_logger.error( "3. Does the subfolder (e.g. 1 2 ...) index of this type exist ?\n" )
        
                    sys.exit()  

                if ( len(os.listdir(match_folder+"/")) ==0):
            
                    self.setup_logger.error( "ERROR: No files inside: ",match_folder ) 
        
                    sys.exit() 
        
        return None

    def Mkdir(self):    
        
        job_folder = self.jobid + "/"

        working_folders = job_folder + self.predict_folder
    
        if ( os.path.isdir(job_folder) and self.overwrite==False): 

            self.setup_logger.error( "%s exists! set 'overwrite=True' upon instantiation "%job_folder ) 

            sys.exit()

        elif ( os.path.isdir(job_folder) and self.overwrite==True ):
        
            self.setup_logger.error( "overwrite the existing folder: %s ! "%job_folder ) 

            os.system("rm -r %s"%job_folder) 

            time.sleep(1)   
            
        os.mkdir(job_folder)

        os.mkdir(working_folders)

        os.mkdir(job_folder + "Restart") 

        os.mkdir(job_folder + "Output")

        self.predict_folder= [ ] 

        self.ref_folder_list = [] 

        for everytype in self.matching_type: 
    
            self.job_folders = working_folders + "/" + everytype + "/" 
    
            os.mkdir(self.job_folders)  

            self.predict_folder.append(self.job_folders)  
        
            self.ref_folder_list.append(self.ref_folder + "/" + everytype ) 
    
        return None 

    def Go_to_Subfolders(self,sub_folders): 

        folders = next(os.walk('.'))[1] 

        if ( folders ): 

            for folder in folders: 

                os.chdir(folder) 

                self.Go_to_Subfolders(sub_folders) 

                os.chdir("../")

        else: 

            sub_folders.append(os.getcwd()) 

        return sub_folders
    
    def Get_Path(self,wk_folder): 

        os.chdir(wk_folder) 

        sub_folders = []  

        sub_folders = self.Go_to_Subfolders(sub_folders) 

        num_folders = len(sub_folders)      

        os.chdir(self.home) 

        return num_folders,sub_folders 

    def Get_Folders_Path(self,main_folders,sub_folder_type):  

        job_address = { } 

        for indx,folder in enumerate(main_folders): 

            sub_index_type = {} 
                            
            for sub_indx in sub_folder_type[self.matching_type[indx]]:  

                num_jobs,sub_folders = self.Get_Path(folder + "/" + sub_indx) 

                sub_folders.sort() 

                sub_index_type[sub_indx] = sub_folders  
    
                job_address[self.matching_type[indx]] = sub_index_type 

        return job_address

    def Transfer_Prepsystem_To_Predict_Folder(self): 

        for indx,matching_type in enumerate(self.matching_type): 

            for sub_indx in self.subfolders_per_type[matching_type]:  

                subprocess.call( "cp"+ " -r " 
                                + self.prep_folder + "/" + "%s/%s" %( matching_type,sub_indx )
                                +  " " + self.predict_folder[indx], shell=True ) 

        return None 

    def Finish(self): 

        self.Transfer_Prepsystem_To_Predict_Folder()

        ref_job_address = self.Get_Folders_Path( self.ref_folder_list,self.subfolders_per_type )    

        predict_job_address = self.Get_Folders_Path( self.predict_folder,self.subfolders_per_type)

        self.setup_logger.debug("Finish setting up the working folder ... ") 

        return self.home,ref_job_address,predict_job_address

