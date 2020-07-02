##############################################################################

# Python-force-field-parameterization-workflow:
# A Python Library for performing force-field optimization

#

# Authors: Jingxiang Guo, Jeremy Palmer

#

# Python-force-field-parameterization-workflow is free software:
# you can redistribute it and/or modify it under the terms of the
# MIT License

# You should have received a copy of the MIT License along with the package.

##############################################################################


"""
This module contains a parent class "set_optimizer" that will be inherited
by other optimizer modules: e.g. gradient_free.py or gradient.py, which may 
contains classes of optimization algorithm.

The purposes of this module are two fold:

1. provide a general interface for parsing a input file for optimization
setting. Initialize, and check the user-provided parameters
from the input file and pass them to other classes through inheritance.

2. provide some utility functions for the output of the restart information,
optimized parameters, and objective functions. These methods will also be
inherited by other classes.

3. provide some utility functions for a optimizer to perform operation like 
bounds, constraints, grouping of optimized parameters.

"""

# Python standard library:
import numpy as np
import logging
import sys
import os

# Local library:
import IO.input_file
import IO.check_type

# Third-party libraries:


class set_optimizer:

    """A Parent class inherited by other optimization classes to have
       a unified interface of parsing input file and perform some general
       operations like outputing and parameters manipulations
 
    Available Parameters 
    --------------------

    self.dump_para : np.ndarray, np.int
        an array of integer that specify how frequent to dump
        the restart file and best parameters.
    self.ptype_lst : list 
        a string defining the type of guess parameters
    self.guess_parameters : np.ndarray, np.float64
        an array of floating point value provided by a user as it is
    self.fit_and_fix : np.ndarray, np.int
        an array determining which parameters to be fixed or fitted
        1: fitted; 0: fixed
    self.bounds_index : np.ndarray, np.int
        an array of index value determine which guess parameters
        to be constrained.
        e.g. 1 means 1st guess parameters, 10 means 10th guess parameters
    self.bounds_fit_index : np.ndarray, np.int
        an array of index value determine which guess parameters
        to be constrained taking into account the fitted parameters
        The fixed parameters can not be constrained. So, the index 
    self.bounds_range : np.ndarray, np.float64
        an array of values size 2 with lower and upper 
    self.optimizer_type : str
        what optimizer selected by the user; currently only
        Nelder-Mead simplex is supported.
    self.optimizer_argument : str 
        The argument associated with the selected optimizer
    self.optimizer_input :
        The optimizer input: input values used by optimizers (optional)

    Available Methods 
    -----------------

    self.group_fixed()
        recombine the fitted parameters with the fixed parameters into
        an array of full parameters.return: full_parameters
        (same length as initial guess)
    self.constrain()
        perform constrain operations on selected fitted parameters within
        the specified bounds
    self.write_optimizer_output()
        a general method to output parameters information to certain
        provided path. Different modes can be used as well
    self.print_optimizer_log():
        print the parsed information from the input file and dump them
        to a log file.
    self.optimizer_restart_content()
        a general method to create a dictionary which contains the
        optimization settings so that the restart file can be used 
     

    Notes
    -----

    This class is best used by other classes through inheritance
    Thus, make sure no conflicts of file or method name in
    the child class

    """

    def __init__(self,
                 input_file,
                 logname=None,
                 skipped=None,
                 stop_after=None):

        # create logger for the optimization jobs

        self.add_logger(logname)

        # parse the input file

        # since restart file requires all the content,
        # read the content before skipped
        self.non_optimizer_data_dict = IO.input_file.parse(input_file,
                                                           0,
                                                           skipped,
                                                           comment=True)

        # read the conetnt for optimizer
        self.input_data_dict = IO.input_file.parse(input_file,
                                                   skipped,
                                                   stop_after)

        self.keys_lst = list(self.input_data_dict.keys())

        self.values_lst = list(self.input_data_dict.values())

        # following the order in input file parse all input values
        # self.pointer: point to current line in the input file
        # self.pointer += 1 for each line read
        # set pointer to the begining

        self.pointer = 0

        self.parse_dumping_freq()

        self.parse_guess()

        self.parse_fit_and_fixed()

        self.parse_bounds()

        self.parse_termination()

        self.parse_optimizer()

        self.check_input_parameters()

    def add_logger(self, logname):

        if (logname is not None):

            self.logger = logging.getLogger(__name__)

            self.logger.setLevel(logging.INFO)

            fh = logging.FileHandler(logname, mode="w")

            formatter = logging.Formatter("%(message)s")

            fh.setFormatter(formatter)

            self.logger.addHandler(fh)

        else:

            self.logger = logging.getLogger(__name__)

        return None

    # dumping the current best parameters, restart simplex,

    def write_optimizer_output(self,
                               freq,
                               n_iteration,
                               output_address,
                               filename,
                               mode,
                               content_dict):

        if (n_iteration % freq == 0):

            if (output_address is None):

                outputfile = filename

            elif (os.path.isdir(output_address)):

                # os.makedirs(output_address)

                outputfile = os.path.join(output_address, filename)

            self.write_output_to_file(n_iteration,
                                      outputfile,
                                      mode,
                                      content_dict)

        return None

    # write output:
    def write_output_to_file(self, n_iteration, outputfile, mode, content):

        if (mode == "a" and n_iteration == 0):

            open(outputfile, "w").close()

        with open(outputfile, mode) as output:

            for line in content.keys():

                output.write(content[line])

        return None

    def parse_dumping_freq(self):

        dump_freq = self.input_data_dict[self.keys_lst[self.pointer]]

        try:

            self.dump_para = np.array(dump_freq).astype(np.int)

        except (ValueError, TypeError):

            self.logger.info("type and value errors"
                             " in reading dump frequency")

            sys.exit("Check errors in log file !")

        # update pointer: only 1 line is read and so point to next line

        self.set_dump_freq()

        self.pointer += 1

        return None

    def set_dump_freq(self):

        if (self.dump_para.size < 2):

            self.logger.error("ERROR: At least two dump frequency"
                              "arguments are needed: "
                              "One for restart,"
                              "One for optimization parameters ")

            self.output_freq = self.dump_para[0]

            self.restart_freq = self.dump_para[0]

        else:

            self.output_freq = self.dump_para[0]

            self.restart_freq = self.dump_para[1]

        return None

    def parse_guess(self):

        guess_parameters = self.input_data_dict[self.keys_lst[self.pointer]]

        all_parameters = []

        self.ptype_lst = []

        for guess in guess_parameters:

            if (IO.check_type.is_string(guess)):

                self.ptype_lst.append(guess)

                continue

            else:

                all_parameters.append(guess)

        # convert all guess parameters to numpy float 64

        self.guess_parameter = np.array(all_parameters).astype(np.float64)

        # update pointer: only 1 line is read and so point to next line

        self.pointer += 1

        return None

    def parse_fit_and_fixed(self):

        fit_and_fix = self.input_data_dict[self.keys_lst[self.pointer]]

        try:

            self.fit_and_fix = np.array(fit_and_fix).astype(np.float64)

        except (ValueError, TypeError):

            self.logger.info("type and value errors"
                             " in fit_and_fixed variables")

            sys.exit("Check errors in log file !")

        self.fit_index = np.array([i for i, x in enumerate(self.fit_and_fix)
                                   if x == 1],
                                  dtype=np.int)

        self.unfit_index = np.array([i for i, x in enumerate(self.fit_and_fix)
                                     if x == 0],
                                    dtype=np.int)

        self.check_guess_parameters()

        # update pointer: only 1 line is read and so point to next line

        self.pointer += 1

        return None

    def parse_bounds(self):

        bounds = self.input_data_dict[self.keys_lst[self.pointer]]

        num_constraints = int(len(bounds)/3)

        if (bounds[0] == "None" or bounds[0] == "none"):

            self.bounds_index = np.array([])

            self.bounds_fit_index = np.array([])

            self.bounds_range = np.array([])

        else:

            try:
                self.bounds_index = (np.array([bounds[idx*3] for
                                               idx in range(num_constraints)])
                                       .astype(np.int)-1)

                self.bounds_fit_index = (np.zeros(self.bounds_index.size)
                                           .astype(np.int))

                for bindx in range(self.bounds_index.size):

                    num_shift = sum(i < self.bounds_index[bindx]
                                    for i in self.unfit_index)

                    self.bounds_fit_index[bindx] = (self.bounds_index[bindx] -
                                                    num_shift)

                self.bounds_range = [[bounds[3*indx+1], bounds[3*indx+2]]
                                     for indx in
                                     range(num_constraints)]

                self.bounds_range = np.array(self.bounds_range)

            except (ValueError, TypeError):

                self.logger.error("ERROR: Type or Value errors "
                                  "in constraints parameters")

                sys.exit("Check errors in log file !")

            self.check_bounds(bounds)

            self.bounds_range.astype(np.float64)

        self.bounds = bounds

        # update pointer: only 1 line is read and so point to next line

        self.pointer += 1

        return None

    def parse_termination(self):

        optimize_settings = self.input_data_dict[self.keys_lst[self.pointer]]

        try:

            self.max_iteration = int(optimize_settings[0])

            self.para_tol = float(optimize_settings[1])

            self.obj_tol = float(optimize_settings[2])

        except (ValueError, TypeError):

            self.logger.error("ERROR: Termination conditon"
                              "format should be: "
                              "integer,float,float")

            sys.exit("Check errors in log file !")

        # update pointer: only 1 line is read and so point to next line

        self.pointer += 1

        return None

    def parse_optimizer(self):

        optimizer_settings = self.input_data_dict[self.keys_lst[self.pointer]]

        try:

            # specific optimizer used: Nelder-Mead simplex,
            # Levenberg–Marquardt, Gauss-Newton ...

            self.optimizer_type = optimizer_settings[0]

            # oher possible arguments:

            self.optimizer_argument = optimizer_settings[0:]

            # rest of arguments ofr the optimizer:

            self.optimizer_input = self.values_lst[self.pointer+1:]

        except(ValueError, TypeError):

            self.logger.error("ERROR: optimizer needs "
                              "at least one argument !")

            sys.exit("Check errors in log file !")

    def check_input_parameters(self):

        # guess parse input parameters

        # The followings are Mandatory optimization parameters
        # passed to any optimizer !!

        self.attribute_exist("guess_parameter")

        self.attribute_exist("fit_and_fix")

        self.attribute_exist("bounds_fit_index")

        self.attribute_exist("bounds_range")

        self.attribute_exist("max_iteration")

        self.attribute_exist("para_tol")

        self.attribute_exist("obj_tol")

        return None

    def attribute_exist(self, attribute):

        if (not hasattr(self, attribute)):

            self.logger.error('No attribute "%s"'
                              'found in input object' % attribute)

            sys.exit("Check errors in log file")

        return None

    def check_bounds(self, bounds):

        # at least 3 arguments needed if any guess parameters to be constrained

        if (len(bounds) % 3 != 0 or len(bounds) < 3):

            self.logger.error("ERROR: At lesat 3 arguments needed "
                              "to constrain a guess parameter")

            sys.exit("Check errors in log file")

        # constraint indexes can not be more than
        # the number of guess parameters

        for i in range(self.bounds_index.size):

            if (np.amax(self.bounds_index) >
                    self.guess_parameter.size - 1):

                self.logger.error("ERROR: The value of constraint index"
                                  "should be < number of guess parameters")

                sys.exit("Check errors in log file")

        for cindex in self.bounds_index:

            if (cindex in self.unfit_index):

                self.logger.error("ERROR:"
                                  "constraint index has to be: "
                                  "fitted variable (=1)."
                                  "Fixed variable (=0) "
                                  "can not be constrained")

                sys.exit("Check errors in log file")

        return None

    def check_guess_parameters(self):

        # The number of guess parameter should be equal to fitted + fixed

        if (self.guess_parameter.size != self.fit_and_fix.size):

            self.logger.error("ERROR: The number of guess parameters "
                              "is not equal to the nubmer of "
                              "fitted parameters (=1) "
                              "+ the number of fixed parameters (=0)")

            sys.exit("Check errors in log file")

        # either fit (1) or fixed (0) ; No other number is allowed

        if (np.any(self.fit_and_fix > 1) or np.any(self.fit_and_fix < 0)):

            self.logger.error("ERROR: The fit or fixed parameter "
                              "should only be either 1 or 0")

            sys.exit("Check errors in log file")

        return None

    # apply the bound to the parameters
    def constrain(self, array):

        num_constraints = self.bounds_fit_index.size

        num_criterion = np.size(self.bounds_range, 0)

        if (num_constraints == num_criterion and num_constraints > 0):

            for i in range(num_criterion):

                lower = self.bounds_range[i][0]

                upper = self.bounds_range[i][1]

                bounds_lower_expr = (lower +
                                     "<=" +
                                     str(array[self.bounds_fit_index[i]]))

                bounds_upper_expr = (str(array[self.bounds_fit_index[i]]) +
                                     "<=" +
                                     upper)

                # evaluate the expression: lower bound < para

                if (eval(bounds_lower_expr)):

                    # lower bound is indeed < para

                    pass

                else:

                    self.logger.info("Lower constraints are applied...")
                    self.logger.info("Parameter: " +
                                     str(array[self.bounds_fit_index[i]]) +
                                     "  is constrained to " + str(lower))

                    array[self.bounds_fit_index[i]] = lower

                # evaluate the expression: lower bound < para

                if (eval(bounds_upper_expr)):

                    # lower bound is indeed < para

                    pass

                else:

                    self.logger.info("Upper constraints are applied...")

                    self.logger.info("Parameter: " +
                                     str(array[self.bounds_fit_index[i]]) +
                                     "  is constrained to " + str(upper))

                    array[self.bounds_fit_index[i]] = upper

        return None

    # combined fitted and fixed parameters into full parameters
    # (same length as guess parameters)

    def group_fixed(self, fitted_para):

        fix_index = self.fit_and_fix == 0

        fit_index = self.fit_and_fix == 1

        para_all = np.zeros(self.guess_parameter.size, dtype=np.float64)

        para_all[fix_index] = self.guess_parameter[fix_index]

        para_all[fit_index] = fitted_para

        return para_all

    def print_optimizer_log(self):

        num_fit = self.guess_parameter[self.fit_and_fix == 1].size

        num_fix = self.guess_parameter[self.fit_and_fix == 0].size

        self.logger.info("\n")
        self.logger.info("----------------------- Initialize Optimization "
                         "Input Parameters -------------------------------\n")

        self.logger.info("Optimizer: %s \n" % self.optimizer_type)

        self.logger.info(102*"-" + "\n")

        self.logger.info("Number of Vertices: %d \n" % (num_fit + 1))
        self.logger.info(102*"-" + "\n")

        self.logger.info("Guess parameters are: \n ")

        self.logger.info(" ".join(str(para)
                                  for para in self.guess_parameter) + "\n")

        self.logger.info(102*"-" + "\n")

        self.logger.info("%d Fitting parameters are:  \n" % (num_fit))

        self.logger.info(" ".join(str(para) for para in
                                  self.guess_parameter[self.fit_index]) +
                         "\n")

        self.logger.info(102*"-" + "\n")

        self.logger.info("%d Fixed parameters are: \n" % (num_fix))
        self.logger.info(" ".join(str(para) for para in
                                  self.guess_parameter[self.unfit_index]) +
                         "\n")

        self.logger.info(102*"-" + "\n")

        self.logger.info("%d constrained parameters : \n"
                         % (self.bounds_index.size))

        for i in range(self.bounds_index.size):

            self.logger.info("The guess parameter: %.6f is constrained "
                             "between %.6f and %.6f "
                             % (float(self.guess_parameter[
                                      self.bounds_index[i]]),
                                float(self.bounds_range[i][0]),
                                float(self.bounds_range[i][1])) +
                             "\n")

        self.logger.info(102*"-" + "\n")

        self.logger.info(" ".join(str(i) for i in
                                  self.optimizer_argument) +
                         "\n")

        return None

    def non_optimizer_restart_content(self, itera):

        content = {}

        content[0] = ("# Current iteration %d: "
                      "This is a restart file \n\n" % itera)

        for i, line in enumerate(self.non_optimizer_data_dict.keys()):

            content[i+1] = (" ".join(para for para in
                                     self.non_optimizer_data_dict[line]) +
                            "\n")

        return content, i

    def optimizer_restart_content(self, itera, best_para):

        # content of restart file following the same as the input file
        # dictionary: keys: line number, values: content of restart file

        # write only general options:
        content, total_lines = self.non_optimizer_restart_content(itera)
        content[total_lines+1] = "# output and restart frequency \n\n"
        content[total_lines+2] = (" ".join(str(para) for para in
                                           self.dump_para) +
                                  "\n\n")

        content[total_lines+3] = "# This is the guess parameters: \n\n"

        content[total_lines+4] = ("# " + " ".join(self.ptype_lst) +
                                  " ".join(str(ele) for ele in
                                           self.guess_parameter) +
                                  "\n\n")
        content[total_lines+5] = "# This is the current best parameters: \n\n"
        content[total_lines+6] = (" ".join(self.ptype_lst) + " " +
                                  " ".join(str(para) for para in best_para) +
                                  "\n\n")

        content[total_lines+7] = "# fit (1) and fix (0) parameters: \n\n"

        content[total_lines+8] = (" ".join(str(para) for para in
                                           self.fit_and_fix) + "\n\n")

        content[total_lines+9] = ("# constraints (index lower-bound "
                                  "upper-bound ... ):\n\n")

        content[total_lines+10] = (" ".join(str(para) for para in
                                            self.bounds) + "\n\n")

        content[total_lines+11] = ("# set termination criterion: "
                                   "max number of iteration, tolerance "
                                   "for parameters, "
                                   "tolerance for objective \n\n")

        content[total_lines+12] = ("%d " % self.max_iteration +
                                   " " +
                                   "{0:.1e}".format(self.obj_tol) +
                                   " " +
                                   "{0:.1e}".format(self.para_tol) +
                                   "\n\n")

        content[total_lines+13] = ("# create (Perturb) or use existing"
                                   " vertices (Restart): \n\n")

        content[total_lines+14] = ("# " + self.optimizer_type +
                                   " " +
                                   " ".join(para for para in
                                            self.optimizer_argument) +
                                   "\n\n")

        content[total_lines+15] = ("# " +
                                   " ".join(str(para) for para in
                                            self.optimizer_input) +
                                   "\n\n")

        content[total_lines+16] = self.optimizer_type + " Restart \n\n"

        return content

    def dump_restart(self,
                     itera,
                     filename_lst,
                     output_address,
                     restart_content_dict):

        self.write_optimizer_output(self.restart_freq,
                                    itera,
                                    output_address,
                                    filename_lst[0],
                                    "a",
                                    restart_content_dict)

        self.write_optimizer_output(self.restart_freq,
                                    itera,
                                    output_address,
                                    filename_lst[1],
                                    "w",
                                    restart_content_dict)

        return None

    def dump_best_objective(self, itera, filename, output_address, best_value):

        best_objective = {}

        best_objective[0] = str(best_value) + "\n"

        self.write_optimizer_output(self.output_freq,
                                    itera,
                                    output_address,
                                    filename,
                                    "a",
                                    best_objective)

        return None

    def dump_best_parameters(self,
                             itera,
                             filename,
                             output_address,
                             best_parameters):

        best_para = {}

        best_para[0] = " ".join(str(para) for para in best_parameters)

        self.write_optimizer_output(self.output_freq,
                                    itera,
                                    output_address,
                                    filename,
                                    "w",
                                    best_para)
        return None
