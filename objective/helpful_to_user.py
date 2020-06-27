# Python standard library:
import numpy as np 
import sys 
import logging
import time 
import os 
import shutil

# local library:  
import IO.check_file 
import IO.check_type
import IO.reader 
import IO.user_provided 

# Third-party library: 

class useful_tools:

    max_total_wait_time = 300

    def __init__(self,
                 matching_type_lst,
                 properties_file_lst,
                 predit_address_tple,
                 argument_tple,
                 output_folder):
        
        self.predicted_and_output_data(matching_type_lst,
                                       properties_file_lst,
                                       predit_address_tple, 
                                       output_folder)         

         
        self.parse_argument_dict(argument_tple)  

        self.check_cores(self.num_cores_predict, len(predit_address_tple))

        self.logger = logging.getLogger(__name__) 
    
        return None
    
    @staticmethod
    def get_folders_level(full_address, matching_type, all_levels):

        """Recursively obtain all names of sub-folders into a list.
           The search stop when the highest level: matching_type name is hit
           These names are concatenated to create a output file name
        """

        upper_level, now = os.path.split(full_address)

        all_levels.append(now)
        
        if (now == matching_type):

            return all_levels

        else:

            useful_tools.get_folders_level(upper_level,matching_type,all_levels)

        return None

    @staticmethod
    def output_files_name(predict_address, matching_type):

        all_levels = []

        useful_tools.get_folders_level(predict_address, matching_type, all_levels)

        output_file = "_".join(each_folder for each_folder in all_levels)

        return output_file

    @staticmethod
    def interm_output_file(matching_type, output_file):

        best_file = output_file + "_best" + ".%s" % matching_type  

        guess_file = output_file + "_guess" + ".%s" % matching_type  

        old_file = output_file + "_old" + ".%s" % matching_type  

        return best_file, guess_file, old_file

    def predicted_and_output_data(self,
                                  matching_type_lst,
                                  properties_file_lst,
                                  predit_address_tple,
                                  output_folder): 

        self.predict_file_lst = []

        self.guess_file_lst = []

        self.old_file_lst = []

        self.best_file_lst = []

        
        for predict_traj in predit_address_tple:
       
            for each_type,filename in zip(matching_type_lst,
                                          properties_file_lst):
        
                output_file = useful_tools.output_files_name(predict_traj,each_type)

                (best_file,
                 guess_file,
                 old_file) = useful_tools.interm_output_file(each_type,
                                                             output_file)
                guess_path = os.path.join(output_folder,guess_file)
               
                old_path = os.path.join(predict_traj,old_file)

                predict_path = os.path.join(predict_traj,filename)

                best_path = os.path.join(output_folder,best_file)

                self.best_file_lst.append(best_path)

                self.predict_file_lst.append(predict_path)
                
                self.guess_file_lst.append(guess_path) 

                self.old_file_lst.append(old_path) 

        return None
    
    # parse mandatory user-input:
    def parse_argument_dict(self,argument):  

        # argument is a tuple
    
        self.sub_folder = argument[0]
        
        # convert objective weight into float 
        
        self.obj_weight = float(argument[1]) 

        # convert cores for analysis into integer

        self.num_cores = int(argument[3]) 

        # equally assign cores for processing predicted and reference data

        self.num_cores_ref = int(self.num_cores/2) 

        self.num_cores_predict = int(self.num_cores/2) 

        return None 

    def check_cores(self,num_cores,num_jobs): 

        # equally assign cores for each job: 
        # if total cores assigned is more than 2,
        # suggesting the use of parallelism   
       
        if (num_cores>=2):   

            if (num_cores%num_jobs ==0): 

                pass  

            else: 

                self.logger.info("ERROR: In isobar matching, total  number of cores for analysis must be divisiable by number of jobs") 

                sys.exit("Check errors in the log file")

        return None 

    def parse_termination(self,argument_str):
   
        keyword_index = IO.user_provided.keyword_exists(argument_str,"t") 

        if ( keyword_index < 0 ):  

            self.logger.warn("WARNNING: missing sampling termination 't'"
                             "in the rdf matching argument;"
                             "dcd file will be opened right after the sampling finishes") 

            self.terminate_crit = 0 

            return None 

        try: 

            self.terminate_crit = int(argument_str[keyword_index+1])  

        except (ValueError,TypeError):

            self.logger.error("ERROR: sampling termination  argument error; The format is: 't 2000' ") 

            sys.exit("Check errors in the log file") 

        return None  

    # Use DCD trajectory to determine the termination criterion 
    def dcd_data_is_avaliable(self, ter_cond, traj_lst):
       
        # check the if number configurations in dcd file is the same as
        # number of configurations given by the user:

        # for each trajectory in the dcd file:
        wait_time_interval = 1 # check data every 1s 

        wait_time = 0 

        for traj_address in traj_lst: 

            if (not IO.check_file.status_is_ok(traj_address)): 

                self.logger.error("WARNNING: The dcd trajecotry: %s "
                                  "is not available for determining"
                                  "termination status! " % traj_address)

                continue

            while True: 

                total_frames, total_atoms = IO.reader.call_read_dcd_header(traj_address)
                
                if (total_frames == ter_cond): 

                    break

                elif (total_frames < ter_cond 
                      and wait_time <= useful_tools.max_total_wait_time):  

                    time.sleep(wait_time_interval) 

                    wait_time += wait_time_interval  

                    self.logger.info("Waiting for the dcd data ... Time elapsed: %d"%wait_time)

                    if ( wait_time == useful_tools.max_total_wait_time):  

                        self.logger.error("Have been waiting for the dcd data" 
                                          "for %d seconds: current frame in the trajectory is: %d "  
                                          "Required number of frames for temrination is: %d "%(wait_time,total_frames,ter_cond))
                        self.logger.error("Exit the program !") 
                        sys.exit("Check the errors in the log file")

                        break 

                elif (total_frames > ter_cond):

                    self.logger.warn("WARNNING: In isobar matching,"
                                     "The number configuration in dcd file set for termination "
                                     "is smaller than that of "
                                     "predicted dcd file")
                    break

        return None

    def rename(self, status) :

        for guess, old, predict in zip(self.guess_file_lst,
                                       self.old_file_lst,
                                       self.predict_file_lst):
    
            if (status == "guess"):

                shutil.move(predict,guess)

            elif (status == "old"):

                shutil.copyfile(predict,old)
                
        return None

    def update(self, status):

        for old, predict, best in zip(self.old_file_lst,
                                      self.predict_file_lst,
                                      self.best_file_lst):

            if (status == "new"):
            
                shutil.move(predict,best)
 
            elif (status == "old"):

                shutil.move(old,best)

        return None

