# Python standard library
import numpy as np
import multiprocessing as mp
import os
import sys
import logging

# Local library:
import IO.reader
import IO.check_file
from objective.rdf_matching.calc_pair_correlation import \
                                                  RadialDistribution as RDF

from objective.helpful_to_user import useful_tools

# Third-party library:

class load(useful_tools):

    def __init__(self,
                 ref_address_tple,
                 predit_address_tple,
                 argument_tple,
                 output_folder):

        self.logger = logging.getLogger(__name__)

        #set the file name

        self.loaded_filename()

        super().__init__(self.matching_type_lst,
                         self.properties_file_lst,
                         predit_address_tple,
                         argument_tple,
                         output_folder)

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

        self.matching_type_lst = ["rdf"]

        self.properties_file_lst = [self.predict_gr]
    
        return None

    def compute_bins_pos(self):

        interval = self.cutoff/self.num_bins

        self.bins_pos = RDF.compute_rdf_bins(interval,self.num_bins)

        return None

    def set_file_address_and_check_status(self, ref_address_tple, predit_address_tple):

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

    def compute_gr_matching_norm(self, gr, interval, bins_pos):
    
        return np.sum((interval*bins_pos*(gr - 1))**2)

    def parse_user_defined(self, argument):

        argument_str = argument[-1].split()

        self.parse_buffersize_arg(argument_str)

        self.parse_cutoff(argument_str)

        self.parse_bins(argument_str)

        return None

    def parse_buffersize_arg(self, argument_str):

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

    def parse_cutoff(self, argument_str):

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

    def parse_bins(self, argument_str):

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

    def optimize(self):
       
        self.dcd_data_is_avaliable(self.terminate_crit,
                                   self.predict_traj_lst )

        sum_sqr_gr = 0

        interval = self.cutoff/self.num_bins

        for ref_gr,predict_traj,predict_gr_path in zip(self.ref_data_lst,
                                                       self.predict_traj_lst,
                                                       self.predict_rdf_path_lst):

            # initialize gr compuation
            gr_calc = RDF(predict_traj,
                          self.num_cores_predict,
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
        
        return self.obj_weight*sum_sqr_gr

