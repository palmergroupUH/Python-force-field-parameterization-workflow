# Python standard library
import numpy as np
import multiprocessing as mp
import os
import sys
import time
import logging

# Local library:
import IO.reader
import IO.check_file
import IO.check_type
from objective.helpful_to_user import useful_tools

# Third-parties libary:


class load(useful_tools):

    objective_type = "isobar"

    def __init__(self,
                 ref_address_tple,
                 predit_address_tple,
                 argument_tple,
                 output_folder):

        self.logger = logging.getLogger(__name__)

        # set the output file type for specific properties generated
        # for rdf it is ".rdf" and for isobar matching it is ".density"
        # These files will be renamed or moved during optimization so that
        # the best predicted properties are always available

        # define the file name

        self.loaded_filename()

        # initialize inherited class: helpful tools
        # this class is used to
        # 1. Check the status of the file
        # 2. track and update the properties
        super().__init__(predit_address_tple,
                         argument_tple,
                         output_folder)

        self.parse_user_defined(argument_tple)

        self.set_file_address_and_check_status(ref_address_tple,
                                               predit_address_tple)

        # compute normalization or data in advance

        self.initialize()

        # use the inherited track_and_update routine to update
        # best predicted properties
        self.track_and_update(load.objective_type,
                              self.ext_type_lst,
                              self.properties_file_lst,
                              predit_address_tple,
                              output_folder)
        return None

    def loaded_filename(self):

        # default reference file name to be used:

        self.ref_density = "Ref.density"

        self.ref_traj = "traj.dcd"

        self.predict_traj = "traj.dcd"

        self.predict_density = "predict.density"

        self.ext_type_lst = ["isobar"]

        self.properties_file_lst = [self.predict_density]

        return None

    def set_file_address_and_check_status(self,
                                          ref_address_tple,
                                          predit_address_tple):

        self.ref_dens_file_lst = []

        self.predict_dens_path_lst = []

        self.predict_traj_lst = []

        self.predict_address_lst = []

        self.ref_data_lst = []

        # for performance, use a generic name
        combine_address = os.path.join

        for ref_address, predict_address in zip(ref_address_tple,
                                                predit_address_tple):

            # generate reference density file name:
            ref_density_file = combine_address(ref_address, self.ref_density)

            # generate predicted trajectory:
            predict_dens_traj = combine_address(predict_address,
                                                self.predict_traj)

            # generate predicted density file path:
            predict_density_path = combine_address(predict_address,
                                                   self.predict_density)

            # check the reference data status:
            IO.check_file.status_is_ok(ref_density_file)

            # save reference density file into list:
            self.ref_dens_file_lst.append(ref_density_file)

            # save the predicted address into a list
            self.predict_address_lst.append(predict_address)

            # save predicted density file into list:
            self.predict_dens_path_lst.append(predict_density_path)

            # save dcd traj file into a list
            self.predict_traj_lst.append(predict_dens_traj)

            # load the reference data in advance

            ref_dens_data = np.average(np.loadtxt(ref_density_file))

            # save data dict in a list
            self.ref_data_lst.append(ref_dens_data)

        return None

    def parse_user_defined(self, argument):

        argument_str = argument[-1].split()

        self.parse_T_arg(argument_str)

        self.parse_tol(argument_str)

        return None

    def parse_T_arg(self, argument_str):

        keyword_index = IO.user_provided.keyword_exists(argument_str, "Temp")

        if (keyword_index < 0):

            self.logger.error("ERROR: missing Temperature 'Temp' "
                              "in the isobar matching argument")

            sys.exit("Check errors in the log file")

        try:

            self.T_ary = self.extract_T(argument_str[keyword_index+1:])

        except (ValueError, TypeError):

            self.logger.error("ERROR: T temperature error in isobar matching;"
                              " The format is 'T 230 240' ")

            sys.exit("Check errors in the log file")

        return None

    def extract_T(self, arg):

        T_ary = np.array([float(T) for T in arg if IO.check_type.is_float(T)])

        return T_ary

    def parse_tol(self, argument_str):

        keyword_index = IO.user_provided.keyword_exists(argument_str, "tol")

        if (keyword_index < 0):

            self.logger.error("ERROR: penalty tolerance 'tol' "
                              "in the isobar matching argument")

            sys.exit("Check errors in the log file")

        try:

            self.tol = int(argument_str[keyword_index+1])

        except (ValueError, TypeError):

            self.logger.error("ERROR: argument error in isobar "
                              "matching; The format is 'tol 2' ")

            sys.exit("Check errors in the log file")

        return None

    def initialize(self):

        # compute penalty parameters

        self.num_T = len(self.ref_data_lst)

        self.ref_density_ary = np.array(self.ref_data_lst)

        self.T_ref_sort = self.sort_temperature(self.ref_density_ary)

        # compute normalization constant from Reference data

        self.normalization = np.var(self.ref_density_ary)

        return None

    def penalty(self, N_tol, num_T, predict_density):

        T_predict_sort = self.sort_temperature(predict_density)

        N_match = 0

        for predict_T, ref_T in zip(T_predict_sort, self.T_ref_sort):

            if (predict_T == ref_T):

                N_match += 1

        return max(num_T - N_tol - N_match, 0)

    def sort_temperature(self, predict_density):

        indx = np.argsort(predict_density)

        return self.T_ary[indx]

    def compute_density(self, filename, keyword):

        if (keyword == "txt"):

            num_lines, num_cols = IO.reader.get_lines_columns(filename)

            data = IO.reader.loadtxt(filename,
                                     num_lines,
                                     num_cols,
                                     skiprows=1,
                                     return_numpy=True)

            return np.average(data)

        elif (keyword == "dcd"):

            return None

    def optimize(self):

        self.dcd_data_is_avaliable(self.terminate_crit,
                                   self.predict_traj_lst)

        sum_sqr_dens = 0

        predict_density_ary = np.zeros(self.num_T)

        for i, (ref_density, predict_density_file) in enumerate(zip(
                                                      self.ref_density_ary,
                                                      self.predict_dens_path_lst)):

            # compute the reference density:

            predict_density = self.compute_density(predict_density_file,
                                                   keyword="txt")

            sum_sqr_dens += (predict_density - ref_density)**2

            predict_density_ary[i] = predict_density

        # normalize the sum of squared errors
        sum_sqr_dens /= self.normalization

        penalty = self.penalty(self.tol, self.num_T, predict_density_ary)

        self.logger.info(30*"--" + "\n\n")
        self.logger.info("Isobar matching Penalty: %d " % penalty)
        self.logger.info(30*"--" + "\n\n")

        return self.obj_weight*(sum_sqr_dens*penalty + sum_sqr_dens)
