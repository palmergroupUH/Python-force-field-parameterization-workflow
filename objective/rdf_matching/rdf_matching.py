# Python standard library 
import numpy as np 
import multiprocessing as mp 
import os 
import sys 
import itertools 
import ctypes 
import time 
import logging 
import shutil
from ctypes import CDLL, POINTER, c_int, c_double,c_char_p,c_long,c_float,byref 

# Local library: 
import IO.reader 
import IO.check_file 
from objective.rdf_matching import calc_pair_correlation

# Third-party library: 


class load():

    max_total_wait_time = 300 # 300s for data  

    def __init__(self,ref_address_tple,predit_address_tple,argument_tple): 

        self.logger = logging.getLogger(__name__)   
    
        #set the file name 

        self.loaded_filename() 
        
        self.parse_argument_dict(argument_tple,len(ref_address_tple))

        self.parse_user_defined(argument_tple) 

        self.set_file_address_and_check_status(ref_address_tple,predit_address_tple)

        self.compute_bins_pos()

        return None 

    def loaded_filename(self): 

        # default reference file name to be used:
         
        self.ref_gr_data = "Ref.gr"

        self.ref_traj = "traj.dcd"

        self.predict_traj = "traj.dcd"
 
        self.predict_gr = "predict.gr"   
    
        return None 

    def compute_bins_pos(self):  

        interval = self.cutoff/self.num_bins

        self.bins_pos = calc_pair_correlation.RadialDistribution.compute_rdf_bins(interval,self.num_bins)

        return None   

    def set_file_address_and_check_status(self,ref_address_tple,predit_address_tple): 

        self.ref_rdf_file_lst = [] 
    
        self.predict_rdf_path_lst = [] 

        self.predict_traj_lst = [] 

        self.predict_address_lst = [] 

        self.ref_data_lst = [] 

        for ref_address,predict_address in zip(ref_address_tple,predit_address_tple): 

            ref_rdf_file = os.path.join(ref_address,self.ref_gr_data)  

            predict_rdf_traj = os.path.join(predict_address,self.predict_traj)

            predict_gr_path = os.path.join(predict_address,self.predict_gr)

            IO.check_file.status_is_ok(ref_rdf_file) 

            self.ref_rdf_file_lst.append(ref_rdf_file) 
    
            self.predict_address_lst.append(predict_address)
    
            self.predict_rdf_path_lst.append(predict_gr_path)  

            self.predict_traj_lst.append(predict_rdf_traj)

            ref_gr_data_dict = self.load_ref_gr_data(ref_rdf_file)

            self.ref_data_lst.append(ref_gr_data_dict) 
    
        return None             

    def load_ref_gr_data(self,filename):

        """ 
        The Ref gr data has the following format: 
    
               column 1,  column 2  
        row 1: bin_pos 1, gr at bin_pos 1 
        row 2: bin_pos 2, gr at bin_pos 2 
        row 3: bin_pos 3, gr at bin_pos 3 
        row 4: bin_pos 4, gr at bin_pos 4 
        ....
        """

        ref_gr_data_dict = {} 

        num_bins,num_column = IO.reader.get_lines_columns(filename) 
        
        ref_gr_data = IO.reader.loadtxt(filename,
                                        num_bins,
                                        num_column,
                                        skiprows=0,
                                        return_numpy=True) 

        ref_bins_pos = ref_gr_data[:,0] 

        ref_gr_data_dict["pos"] = ref_bins_pos  

        ref_gr_data_dict["gr"] = ref_gr_data[:,1] 

        ref_gr_data_dict["num_bins"] = ref_bins_pos.size 

        ref_gr_data_dict["interval"] = (ref_bins_pos[1] - ref_bins_pos[0])

        ref_gr_data_dict["norm"] = self.compute_gr_matching_norm(
                                                       ref_gr_data_dict["gr"],
                                                       ref_gr_data_dict["interval"],
                                                       ref_gr_data_dict["pos"])
    
        return ref_gr_data_dict   

    def compute_gr_matching_norm(self,gr,interval,bins_pos): 
    
        return np.sum((interval*bins_pos*(gr - 1))**2) 

    def parse_argument_dict(self,argument,num_jobs): 

        # argument is a tuple 

        self.sub_folder = argument[0] 

        # convert objective weight into float 

        self.obj_weight = float(argument[1]) 
        
        # convert cores for analysis into integer 

        self.num_cores = int(argument[3] ) 

        self.parse_cores(num_jobs)

        return None 

    def parse_user_defined(self,argument): 

        argument_str = argument[-1].split() 

        self.parse_buffersize_arg(argument_str)  

        self.parse_cutoff(argument_str) 

        self.parse_bins(argument_str) 

        self.parse_termination(argument_str) 

        return None 

    def parse_buffersize_arg(self,argument_str): 

        keyword_index = IO.user_provided.keyword_exists(argument_str,"bf") 

        if ( keyword_index < 0 ):  

            self.logger.error("ERROR: missing buffersize 'bf' in the force matching argument") 

            sys.exit("Check errors in the log file") 

        try: 

            self.buffersize = int(argument_str[keyword_index+1])  

        except ( ValueError,TypeError):

            self.logger.error("ERROR: buffer index argument error; The format is 'bf integer' ") 

            sys.exit("Check errors in the log file") 

        return None 

    def parse_cores(self,num_jobs): 

        # equally assign cores for each job: 

        if (self.num_cores%num_jobs ==0): 

            self.cores_per_job = int(self.num_cores/num_jobs) 

        else: 

            self.logger.info("ERROR: In RDF matching, total  number of cores for analysis must be divisiable by number of jobs") 

            sys.exit("Check errors in the log file")

        return None 

    def parse_cutoff(self,argument_str):   

        keyword_index = IO.user_provided.keyword_exists(argument_str,"c") 

        if ( keyword_index < 0 ):  

            self.logger.error("ERROR: missing cutoff 'c' in the rdf matching argument") 

            sys.exit("Check errors in the log file") 

        try: 

            self.cutoff = float(argument_str[keyword_index+1])  

        except (ValueError,TypeError):

            self.logger.error("ERROR: cutoff argument error; The format is: 'c 3.8' ") 

            sys.exit("Check errors in the log file") 

        return None 

    def parse_bins(self,argument_str): 

        keyword_index = IO.user_provided.keyword_exists(argument_str,"b") 

        if ( keyword_index < 0 ):  

            self.logger.error("ERROR: missing number of bins 'b' in the rdf matching argument") 

            sys.exit("Check errors in the log file") 

        try: 

            self.num_bins = int(argument_str[keyword_index+1])  

        except (ValueError,TypeError):

            self.logger.error("ERROR: number of bins argument error; The format is: 'b 200' ") 

            sys.exit("Check errors in the log file") 

        return None 

    def parse_termination(self,argument_str): 
   
        keyword_index = IO.user_provided.keyword_exists(argument_str,"t") 

        if ( keyword_index < 0 ):  

            self.logger.warn("WARNNING: missing sampling termination 't'"
                             "in the rdf matching argument;"
                             "dcd file will be opened right after the sampling finishes") 

            self.sample_termination = 0 

            return None 

        try: 

            self.sample_termination = int(argument_str[keyword_index+1])  

        except (ValueError,TypeError):

            self.logger.error("ERROR: sampling termination  argument error; The format is: 't 2000' ") 

            sys.exit("Check errors in the log file") 

        return None  

    def dcd_data_is_avaliable(self):  
        
        # for each trajectory in the dcd file: 

        wait_time_interval = 1 # check data every 1s 

        wait_time = 0 

        for traj_address in self.predict_traj_lst:  

            if (not IO.check_file.status_is_ok(traj_address)): 

                self.logger.error("ERROR: trajecotry %s does not exist ! "%traj_address) 

                sys.exit("Check errors in the log file")

            while True: 

                total_frames, total_atoms = IO.reader.call_read_dcd_header(traj_address) 

                if (total_frames == self.sample_termination): 

                    self.logger.info("DCD data is ready for RDF matching ...")

                    break

                elif (total_frames < self.sample_termination 
                      and wait_time <= load.max_total_wait_time):  

                    time.sleep(wait_time_interval) 

                    wait_time += wait_time_interval  

                    self.logger.info("Waiting for the dcd data ... Time elapsed: %d"%wait_time)

                    if ( wait_time == load.max_total_wait_time):  

                        self.logger.error("Have been waiting for the dcd data" 
                                          "for %d seconds: current frame in the trajectory is: %d "  
                                          "Required number of frames for temrination is: %d "%(wait_time,total_frames,self.sample_termination))
                        self.logger.error("Exit the program !") 
                        sys.exit("Check the errors in the log file")

                        break 
                   
                elif (total_frames > self.sample_termination): 

                    self.logger.warn("WARNNING: In rdf matching,"
                                      "number of termination configurations "
                                      "is smaller than the "
                                      "total configuration from predicted dcd file"
                                      ";Reset the termination configuration "
                                      "to match the configurations from predicted dcd file")

                    break        

        return None 
    
    def optimize(self):
        
        self.dcd_data_is_avaliable()  

        sum_sqr_gr = 0 

        interval = self.cutoff/self.num_bins 

        for ref_gr,predict_traj,predict_gr_path in zip(self.ref_data_lst,
                                                       self.predict_traj_lst,
                                                       self.predict_rdf_path_lst): 

            # initialize gr compuation
            gr_calc = calc_pair_correlation.RadialDistribution(predict_traj,
                                                          self.cores_per_job,
                                                          self.cutoff,
                                                          self.num_bins,
                                                          self.buffersize)  

            # compute gr in parallel 
            gr_predict = gr_calc.compute() 

            # interpolate the Reference gr
            ref_gr_interp = np.interp(self.bins_pos,
                                      ref_gr["pos"],
                                      ref_gr["gr"]) 

            # write the gr into a file callled "predict.gr" 

            gr_calc.dump_gr(predict_gr_path)
        
            sum_sqr_gr += np.sum(interval*(self.bins_pos*(ref_gr_interp 
                                 - gr_predict))**2)/ref_gr["norm"]  
        

            print ("objective:",sum_sqr_gr)
        return self.obj_weight*sum_sqr_gr 

    def rename(self,status,output_folder):  

        for predict_address in self.predict_address_lst:   
    
            gr_data_path = os.path.join(predict_address,self.predict_gr) 

            if (status == "guess"):

                initia_predicted_rdf = self.sub_folder+"_guess"+".rdf"
        
                # output path of rdf:  

                dest_rdf = os.path.join(output_folder,initia_predicted_rdf)

                # send the file to output:  

                shutil.move(gr_data_path,dest_rdf)     

            elif (status =="old"): 

                old_gr_data = os.path.join(predict_address,status + ".rdf") 

                shutil.copyfile(gr_data_path,old_gr_data)  
                
        return None 

    def update(self,status,output_folder): 

        for predict_address in self.predict_address_lst:  

            best_predicted_rdf = self.sub_folder + "_best" + ".rdf" 
        
            output_best_rdf = os.path.join(output_folder,best_predicted_rdf) 

            gr_data_path = os.path.join(predict_address,self.predict_gr)

            if (status == "new"): 
            
                shutil.move(gr_data_path,output_best_rdf)  
 
            elif (status == "old"): 

                old_gr_data = os.path.join(predict_address,status+".rdf") 

                shutil.move(old_gr_data,output_best_rdf)                  

        return None 



