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
# import constants module: units class, Avogadro number, Boltzman constants
from objective.const_mod import Units, NA, kb

# Third-party library:

# define global variables:

# This defines the maximum size to be loaded into memory during initializatioin


class load(useful_tools):

    """Implmentation of Force matching method   

    Inherited methods and attributes  

    ----------

    A_n : np.ndarray, float

        A_n[n] is nth value of timeseries A.  Length is deduced from vector.

    B_n : np.ndarray, float, optional, default=None

        B_n[n] is nth value of timeseries B.  Length is deduced from vector.

        If supplied, the cross-correlation of timeseries A and B will be estimated instead of the

        autocorrelation of timeseries A.  

    fast : bool, optional, default=False

        f True, will use faster (but less accurate) method to estimate correlation

        time, described in Ref. [1] (default: False).  This is ignored

        when B_n=None and fft=True.

    mintime : int, optional, default=3

        minimum amount of correlation function to compute (default: 3)

        The algorithm terminates after computing the correlation time out to mintime when the

        correlation function first goes negative.  Note that this time may need to be increased

        if there is a strong initial negative peak in the correlation function.

    fft : bool, optional, default=False

        If fft=True and B_n=None, then use the fft based approach, as

        implemented in statisticalInefficiency_fft().



    Returns

    -------

    g : np.ndarray,

        g is the estimated statistical inefficiency (equal to 1 + 2 tau, where tau is the correlation time).

        We enforce g >= 1.0.



    Notes

    -----

    The same timeseries can be used for both A_n and B_n to get the autocorrelation statistical inefficiency.

    The fast method described in Ref [1] is used to compute g.



    References

    ----------

    [1] Ercolessi, F., & Adams, J. B. (1994). 
        Interatomic Potentials from First-Principles Calculations: 
        The Force-Matching Method. Europhysics Letters ({EPL}), 
        26(8), 583–588. https://doi.org/10.1209/0295-5075/26/8/005 

    [2] 
       

    Examples

    --------
    """
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

        # initialize inherited class: helpful tools
        # this class is used to
        # 1. Check the status of the file
        # 2. track and update the properties
        super().__init__(predit_address_tple,
                         argument_tple,
                         output_folder)

        self.set_file_address_and_check_status(ref_address_tple,
                                               predit_address_tple)

        # parse the user-defined input information:

        self.parse_user_defined(argument_tple)

        self.Initialize_energy_matching()

        self.Initialize_force_matching()

        self.Initialize_virial_matching(ref_address_tple,
                                        predit_address_tple)
    
        # use the inherited track_and_update routine to update
        # best predicted properties
        self.track_and_update(load.objective_type,
                              self.ext_type_lst,
                              self.properties_file_lst,
                              predit_address_tple,
                              output_folder)

        return None

    def loaded_filename(self):

        # Define all loaded files name for objective function
        # modify the following file names if needed

        self.Ref_energy_file = "Ref.eng"

        self.Ref_force_file = "Ref.force"

        self.Ref_virial_file = "Ref.virial"

        self.Ref_temp_file = "Ref.temp"
    
        self.Ref_volume_file = "Ref.volume"

        self.predict_energy_file = "predict.eng"

        self.predict_force_file = "predict.force"
            
        self.predict_virial_file = "predict.virial"

        self.ext_type_lst = []

        self.properties_file_lst = []

        return None

    def determine_properties_used(self):

    
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
        # user defined: "w 1.0 1.0 0.0 bf 5000 eng abs virial"
        # get the weight between energy and force

        argument_str = argument[-1].split()

        self.parse_weight_arg(argument_str)

        self.parse_buffersize_arg(argument_str)

        self.parse_eng_arg(argument_str)

        self.parse_virial_arg(argument_str)

        return None

    def parse_weight_arg(self, argument_str):

        key_indx = IO.user_provided.keyword_exists(argument_str, "w")

        if (key_indx < 0):

            self.logger.warn("WARRNING: missing weight 'w' in the "
                             "force matching argument\n"
                             "If none, force and energy is assumed "
                             "to be equally weighted")

            self.weight_force_eng = np.array([1.0, 1.0, 0.0], dtype=np.float64)

            return None

        try:

            self.weight_force_eng = np.array([
                                             float(argument_str[key_indx+1]),
                                             float(argument_str[key_indx+2]),
                                             float(argument_str[key_indx+3])])

        except (ValueError, TypeError):

            self.logger.error("ERROR: type or value errors in choosing "
                              "weight between force,  energy, and virial; "
                              "The format should be 'w float float float' ")

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

        except(ValueError, TypeError):

            self.logger.error("ERROR: energy keyword type error; The "
                              "keyword is a string;'eng abs' or "
                              "'eng relative'' ")

            sys.exit("Check errors in the log file")

        return None

    def parse_virial_arg(self, argument_str):

        key_indx = IO.user_provided.keyword_exists(argument_str, "virial")
        # No virial keyword argument and virial matching weight is 0
        if (key_indx < 0 and self.weight_force_eng[2] ==0):

            return None

        elif (key_indx < 0 and self.weight_force_eng[2] !=0):

           sys.exit("ERROR: virial matching has nonzero weight, "
                    "However, no 'virial' keyword given") 
        
        try:     

            self.ref_dof = float(argument_str[key_indx+1]) 

            self.predict_dof = float(argument_str[key_indx+2])

        except(ValueError, TypeError):

            self.logger.error("ERROR: virial keyword type error; " 
                              "the keyword format 'Ref Dof, predict DoF'")

            sys.exit("Check errors in the log file")

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

        self.ext_type_lst.extend(["force"]) 

        self.properties_file_lst.extend([self.predict_force_file])

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
#                             Virial Matching:
# ----------------------------------------------------------------------------

    def Initialize_virial_matching(self, ref_address_tple, predit_address_tple):

        # if the weight of virial matching is 0
        # skip the virial matching
        # a list of default virial objective function is 0
        if (self.weight_force_eng[2] == 0.0):

            self.virial_obj_lst = []
            
            for ref_address, predict_address in zip(ref_address_tple,
                                                    predit_address_tple):

                self.virial_obj_lst.append(0)

            return None

        self.ext_type_lst.extend(["virial"])

        self.properties_file_lst.extend([self.predict_virial_file])

        self.ref_virial_path_lst = []
    
        self.ref_volume_path_lst = []

        self.predict_virial_path_lst = []

        self.ref_virial_norm_lst = []
    
        self.ke_loss_lst = []

        units = Units("real")

        self.virial_factor = units.p_scale * units.vol_scale * NA
       
        for ref_address, predict_address in zip(ref_address_tple,
                                                predit_address_tple):
   
            # create *.virial, *.temp files paths 
            ref_virial_path = os.path.join(ref_address, self.Ref_virial_file)

            ref_temp_path = os.path.join(ref_address, self.Ref_temp_file)

            ref_volume_path = os.path.join(ref_address, self.Ref_volume_file)

            predict_virial_path = os.path.join(predict_address,
                                               self.predict_virial_file)

            self.predict_virial_path_lst.append(predict_virial_path)
    
            self.ref_virial_path_lst.append(ref_virial_path)

            # ---------- load data ---------

            ref_virial_data = IO.reader.np_loadtxt(ref_virial_path, skiprows=1)

            T_data = IO.reader.np_loadtxt(ref_temp_path, skiprows=1)

            self.ref_virial_norm_lst.append(np.var(ref_virial_data))

            self.ref_volume_path_lst.append(ref_volume_path)

            ke_loss = self.compute_kinetic_loss(T_data,
                                                self.ref_dof,
                                                self.predict_dof)

            self.ke_loss_lst.append(ke_loss)

        return None

    def compute_kinetic_loss(self, Temp, ref_dof, predict_dof):

        two_ke_delta = (ref_dof - predict_dof) * Temp * kb * NA

        return two_ke_delta  

    def compute_virial_matching_objective(self):

        # if the weight of virial matching is 0
        # skip the virial matching
        if (self.weight_force_eng[2] == 0.0):

            return None 

        self.virial_obj_lst = []

        for (ref_path,
             predict_path,
             vol_path,
             ke_loss,
             norm) in zip(self.ref_virial_path_lst,
                          self.predict_virial_path_lst,
                          self.ref_volume_path_lst,
                          self.ke_loss_lst,
                          self.ref_virial_norm_lst):
    
            ref_virial = IO.reader.np_loadtxt(ref_path, skiprows=1)

            predict_virial = IO.reader.np_loadtxt(predict_path, skiprows=1)

            volume = IO.reader.np_loadtxt(vol_path, skiprows=1)

            w_ref = ref_virial* self.virial_factor * volume # units of J/mol
             
            w_predict = predict_virial * self.virial_factor * volume
            
            self.virial_obj_lst.append(np.mean((((w_ref - w_predict) + ke_loss)**2)/norm))

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

        # set the properties extension *.eng
        self.ext_type_lst.extend(["eng"]) 

        # track the properties file and updated it as best predicted
        self.properties_file_lst.extend([self.predict_energy_file])

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
#                             Compute overall objective:
# ----------------------------------------------------------------------------

    def optimize(self):

        # inherited function: before evaluating objective functions
        self.check_predicted_data_status(self.ref_force_lines,
                                         self.predict_force_file_lst)

        scaled_eng_objective = 0

        scaled_force_objective = 0
    
        scaled_virial_objective = 0

        self.compute_force_matching_objective()

        self.compute_energy_matching_objective()

        self.compute_virial_matching_objective()

        for e_obj, f_obj, v_obj in zip(self.eng_obj_lst,
                                       self.fm_objective_lst,
                                       self.virial_obj_lst):

            scaled_eng_objective += self.weight_force_eng[0] * e_obj

            scaled_force_objective += self.weight_force_eng[1] * f_obj

            scaled_virial_objective += self.weight_force_eng[2] * v_obj

        return self.obj_weight*(scaled_eng_objective +
                                scaled_force_objective +
                                scaled_virial_objective)
