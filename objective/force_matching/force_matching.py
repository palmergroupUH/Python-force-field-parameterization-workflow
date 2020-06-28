# Python standard library:
import numpy as np
import multiprocessing as mp
import sys
import logging
import os

# local library:
import IO.check_file
import IO.check_type
import IO.reader
import IO.user_provided
from IO.reader import read_LAMMPS_traj_in_parallel as read_LAMMPS
from objective.helpful_to_user import useful_tools

# Third-party library:

# define global variables:

# This defines the maximum size to be loaded into memory during initializatioin


class load(useful_tools):

    count_jobs = 0

    # MB
    total_file_size_allowed = 1000

    objective_type = "force"

    def __init__(self,
                 ref_address_tple,
                 predit_address_tple,
                 argument_tple,
                 output_folder):

        load.count_jobs += 1

        self.logger = logging.getLogger(__name__)

        # load pre-determined file name
        self.loaded_filename()

        super().__init__(load.objective_type,
                         self.ext_type_lst,
                         self.properties_file_lst,
                         predit_address_tple,
                         argument_tple,
                         output_folder)

        self.set_file_address_and_check_status(ref_address_tple,
                                               predit_address_tple)

        # parse the user-defined input information:

        self.parse_user_defined(argument_tple)

        self.Initialize_force_matching()

        self.Initialize_energy_matching()

        return None

    def loaded_filename(self):

        # modify the following file names if needed

        self.Ref_energy_file = "Ref.eng"

        self.Ref_force_file = "Ref.force"

        self.predict_energy_file = "predict.eng"

        self.predict_force_file = "predict.force"

        self.ext_type_lst = ["eng", "force"]

        self.properties_file_lst = [self.predict_energy_file,
                                    self.predict_force_file]

        return None

    def set_file_address_and_check_status(self,
                                          ref_address_tple,
                                          predit_address_tple):

        self.Ref_force_file_lst = []

        self.predict_force_file_lst = []

        self.Ref_energy_file_lst = []

        self.predict_energy_file_lst = []

        self.ref_force_lines = []

        self.ref_eng_lines = []

        self.ref_eng_cols = []

        self.predicted_address_lst = []

        for ref_address, predict_address in zip(ref_address_tple,
                                                predit_address_tple):

            # get Reference energy and force address:

            ref_energy_file = os.path.join(ref_address, self.Ref_energy_file)

            ref_force_file = os.path.join(ref_address, self.Ref_force_file)

            predict_energy_file = os.path.join(predict_address,
                                               self.predict_energy_file)

            predict_force_file = os.path.join(predict_address,
                                              self.predict_force_file)

            self.predicted_address_lst.append(predict_address)

            self.Pre_load_energy_data(ref_energy_file)

            IO.check_file.status_is_ok(ref_energy_file)

            IO.check_file.status_is_ok(ref_force_file)

            (num_lines_eng,
             num_colums_eng) = IO.reader.get_lines_columns(ref_energy_file)

            (num_lines_force,
             num_colums_force) = IO.reader.get_lines_columns(ref_force_file)

            self.ref_eng_lines.append(num_lines_eng)

            self.ref_eng_cols.append(num_colums_eng)

            self.ref_force_lines.append(num_lines_force)

            self.Ref_energy_file_lst.append(ref_energy_file)

            self.Ref_force_file_lst.append(ref_force_file)

            self.predict_energy_file_lst.append(predict_energy_file)

            self.predict_force_file_lst.append(predict_force_file)

        return None

    def Pre_load_energy_data(self, file_address):

        # set default preload of Ref force and energy as false:

        self.load_ref_eng = False

        # check Reference energy file is too big or not

        ref_eng_file_size = IO.check_file.get_file_size(file_address,
                                                        units="MB")

        if (ref_eng_file_size < load.total_file_size_allowed):

            self.load_ref_eng = True

        return None

# ----------------------------------------------------------------------------
#                             Parse the input:
# ----------------------------------------------------------------------------

    # parse the user-defined input
    def parse_user_defined(self, argument):

        # --------------- user defined argument ----------------------
        # user defined: "w 1.0 1.0 bf 5000 eng abs virial"
        # get the weight between energy and force

        argument_str = argument[-1].split()

        self.parse_weight_arg(argument_str)

        self.parse_buffersize_arg(argument_str)

        self.parse_eng_arg(argument_str)

        return None

    def parse_weight_arg(self, argument_str):

        key_indx = IO.user_provided.keyword_exists(argument_str, "w")

        if (key_indx < 0):

            self.logger.warn("WARRNING: missing weight 'w' in the "
                             "force matching argument\n"
                             "If none, force and energy is assumed "
                             "to be equally weighted")

            self.weight_force_eng = np.array([1.0, 1.0], dtype=np.float64)

            return None

        try:

            self.weight_force_eng = np.array([
                                             float(argument_str[key_indx+1]),
                                             float(argument_str[key_indx+2])])

        except (ValueError, TypeError):

            self.logger.error("ERROR: type or value errors in choosing "
                              "weight between force and energy; "
                              "The format should be 'w float float' ")

            sys.exit("Check errors in the log file")

            self.logger.warn("WARRNING: missing weight 'w' in the force "
                             "matching argument\n"
                             "If none, force and energy is assumed "
                             "to be equally weighted")

        return None

    def parse_buffersize_arg(self, argument_str):

        key_indx = IO.user_provided.keyword_exists(argument_str, "bf")

        if (key_indx < 0):

            self.logger.error("ERROR: missing buffersize 'bf' in the "
                              "force matching argument")

            sys.exit("Check errors in the log file")

        try:

            self.buffersize = int(argument_str[key_indx+1])

        except (ValueError, TypeError):

            self.logger.error("ERROR: buffer index argument error; "
                              "The format is 'bf integer' ")

            sys.exit("Check errors in the log file")

        return None

    def parse_eng_arg(self, argument_str):

        key_indx = IO.user_provided.keyword_exists(argument_str, "eng")

        if (key_indx < 0):

            self.eng_keyword = "var"

            self.logger.warn("WARRNING: missing engergy matching 'eng' in "
                             "the force matching argument\n")

            self.logger.warn("if none, 'eng relative' is used instead\n")

            return None

        if (not IO.check_type.is_string(argument_str[key_indx+1])):

            self.logger.error("ERROR: energy keyword type error; The keyword "
                              "is a string;'eng abs' or 'eng var'' ")

            sys.exit("Check errors in the log file")

        try:

            self.eng_keyword = argument_str[key_indx+1]

        except (ValueError, TypeError):

            self.logger.error("ERROR: energy keyword type error; The "
                              "keyword is a string;'eng abs' or "
                              "'eng relative'' ")

            sys.exit("Check errors in the log file")

        return None

    def parse_virial_arg(self):

        key_indx = IO.user_provided.keyword_exists(argument_str, "virial")

        if (key_indx < 0):

            self.virial_keword = False

            return None

        return None

    def print_objective_info(self):

        self.logger.info("Reference data address:  \n")
        self.logger.info("The sub_folder name: %s\n" % sub_folder)
        self.logger.info("The weight of objective function : %.3f \n"
                         % weight)
        self.logger.info("Number of cores for running sampling: %d \n"
                         % cores_for_sampling)
        self.logger.info("Number of cores for computing objective: %d\n"
                         % cores_for_objective)
        self.logger.info("The other arbitrary argument: %s \n" % argument)

        return None

# -----------------------------------------------------------------------------
#                             Force Matching:
# -----------------------------------------------------------------------------

    def Initialize_force_matching(self):

        if (self.weight_force_eng[1] == 0.0):

            self.logger.warn("WARNNING: The weight for force matching "
                             "is 0; skip the force matching\n")

            return None

        self.num_congigs_lst = []

        self.num_atoms_lst = []

        self.ref_force_norm_lst = []

        self.workers = mp.Pool(self.num_cores)

        for i, force_file_name in enumerate(self.Ref_force_file_lst):

            num_lines = self.ref_force_lines[i]

            num_atoms = IO.reader.read_LAMMPS_traj_num_atoms(force_file_name)

            self.num_atoms_lst.append(num_atoms)

            # get the number of configurations:
            num_configs = IO.reader.get_num_configs_LAMMPS_traj(num_atoms,
                                                                num_lines)

            self.num_congigs_lst.append(num_configs)

            force_ref_jobs = read_LAMMPS(force_file_name,
                                         self.num_cores,
                                         num_atoms,
                                         num_configs,
                                         1,
                                         self.buffersize,
                                         self.workers)

            # computing the force normalization:

            self.ref_force_norm_lst.append(self.force_norm(force_ref_jobs,
                                                           num_configs,
                                                           num_atoms,
                                                           num_column=3))

        self.workers.close()

        self.workers.join()

        return None

    def force_norm(self,
                   force_job_list,
                   total_configs,
                   num_atoms,
                   num_column):

        sum_refforce = 0

        sqr_ave = 0

        # loop over all cores of reading force data

        for output in force_job_list:

            # get reference data from current core

            Reference_data = output.get()

            sum_refforce = sum_refforce + np.sum(Reference_data)

            sqr_ave = sqr_ave + np.sum(Reference_data*Reference_data)

        average_sqr = (sum_refforce/(total_configs*num_atoms*num_column))**2

        sqr_average = sqr_ave/(total_configs*num_atoms*num_column)

        variances_ref = ((sqr_average - average_sqr) *
                         total_configs *
                         num_atoms *
                         num_column)

        return variances_ref

    def compute_force_matching_objective(self):

        self.fm_objective_lst = []

        i = 0

        for ref_file, predict_file in zip(self.Ref_force_file_lst,
                                          self.predict_force_file_lst):

            if (self.weight_force_eng[1] != 0.0):

                self.ref_workers = mp.Pool(self.num_cores_ref)

                self.predict_workers = mp.Pool(self.num_cores_predict)

                # launch the job in parallel jobs
                # start reading reference force data

                force_ref_jobs = read_LAMMPS(ref_file,
                                             self.num_cores_ref,
                                             self.num_atoms_lst[i],
                                             self.num_congigs_lst[i],
                                             1,
                                             self.buffersize,
                                             self.ref_workers)

                # start reading predicted force data
                force_predict_jobs = read_LAMMPS(predict_file,
                                                 self.num_cores_predict,
                                                 self.num_atoms_lst[i],
                                                 self.num_congigs_lst[i],
                                                 1,
                                                 self.buffersize,
                                                 self.predict_workers)

                sum_sqr_diff = 0

                # update the counter

                for ref_output, predict_output in zip(force_ref_jobs,
                                                      force_predict_jobs):

                    sum_sqr_diff += np.sum(np.square((ref_output.get() -
                                                      predict_output.get())))

                self.fm_objective_lst.append(sum_sqr_diff /
                                             self.ref_force_norm_lst[i])

                i += 1

                self.ref_workers.close()

                self.predict_workers.close()

                self.ref_workers.join()

                self.predict_workers.join()

            else:

                self.fm_objective_lst.append(0)

        return None

# ----------------------------------------------------------------------------
#                             Energy Matching:
# ----------------------------------------------------------------------------

    def Initialize_energy_matching(self):

        # if weight of energy is 0, no need to do energy matching:

        if (self.weight_force_eng[0] == 0.0):

            self.logger.warn("WARNNING: The weight for energy matching "
                             "is 0; skip energy matching\n")

            return None

        self.ref_eng_data_lst = []

        self.ref_eng_norm_lst = []

        for i, ref_eng_file in enumerate(self.Ref_energy_file_lst):

            num_lines = self.ref_eng_lines[i]

            num_cols = self.ref_eng_cols[i]

            ref_energy_data, energy_norm = self.energy_norm(ref_eng_file,
                                                            num_lines,
                                                            num_cols)

            self.ref_eng_data_lst.append(ref_energy_data)

            self.ref_eng_norm_lst.append(energy_norm)

        return None

    def energy_norm(self, Ref_eng_file, num_lines_eng, num_cols):

        if (self.load_ref_eng is True):

            ref_energy_data = IO.reader.loadtxt(Ref_eng_file,
                                                num_lines_eng,
                                                num_cols,
                                                skiprows=0,
                                                return_numpy=True)

            energy_norm = np.var(ref_energy_data)

            return ref_energy_data, energy_norm

    def compute_energy_matching_objective(self):

        self.eng_obj_lst = []

        for i, (ref_file, predict_file) in enumerate(zip(self.Ref_energy_file_lst,
                                                         self.predict_energy_file_lst)):

            if (self.weight_force_eng[0] != 0.0):

                predicted_eng_data = IO.reader.loadtxt(predict_file,
                                                       self.ref_eng_lines[i]+1,
                                                       self.ref_eng_cols[i],
                                                       skiprows=1,
                                                       return_numpy=True)

                if (self.eng_keyword == "var"):

                    self.eng_obj_lst.append(self.scaled_var_energy(predicted_eng_data,
                                                                   self.ref_eng_data_lst[i],
                                                                   self.ref_eng_norm_lst[i]))

                elif (self.eng_keyword == "abs"):

                    self.eng_obj_lst.append(self.scaled_abs_energy(predicted_eng_data,
                                                                   self.ref_eng_data_lst[i],
                                                                   self.ref_eng_norm_lst[i]))

                else:

                    self.logger.info("The energy matching keyword not "
                                     "recognized: Choose 'var' or 'abs'")
                    sys.exit("Check errors in the log file !")

            else:

                self.eng_obj_lst.append(0)

        return None

    def scaled_var_energy(self, predicted_eng, ref_energy, eng_norm):

        diff = predicted_eng - ref_energy

        ave_diff = np.average(diff)

        relative_eng = (diff - ave_diff)**2

        return np.average(relative_eng/eng_norm)

    def scaled_abs_energy(self, predicted_eng, ref_energy, eng_norm):

        return np.average((predicted_eng - ref_energy)**2/eng_norm)


# ----------------------------------------------------------------------------
#                             Virial  Matching
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
#                             Compute overall objective:
# ----------------------------------------------------------------------------

    def optimize(self):

        # before evaluating objective functions
        self.check_predicted_data_status(self.ref_force_lines,
                                         self.predict_force_file_lst)

        eng_weight = self.weight_force_eng[0]

        force_weight = self.weight_force_eng[1]

        scaled_eng_objective = 0

        scaled_force_objective = 0

        self.compute_force_matching_objective()

        self.compute_energy_matching_objective()

        for e_obj, f_obj in zip(self.eng_obj_lst, self.fm_objective_lst):

            scaled_eng_objective += eng_weight * e_obj

            scaled_force_objective += force_weight * f_obj

        return self.obj_weight*(scaled_eng_objective + scaled_force_objective)
