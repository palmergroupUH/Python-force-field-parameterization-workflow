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
This module contains the main function "optimize_main" to run the force-field
optimization.

The executable "optimize" is invoked from the command-line interface. It
will call "main()", which then call the function "optimize_main".
Some other command-line programs related to this package can be developed,
and invoked in an anaglous fashion.


The "optimize_main" is composed of several instances from different modules,
whic are laid out in procedure-oriented fashion so that the user can
easily understand the whole workflow. This should make the customization
more transparant.

"""
# Python standard library:
import numpy as np
import os
import logging
import sys
import importlib
# Local library:
import IO
import objective
# Third-party libraries:


class prepare():

    def __init__(self,
                 JOBID,
                 ref_dict,
                 predict_dict,
                 arg_dict,
                 sampling_method):

        self.output_folder = os.path.join(JOBID, "Output")

        self.logger = logging.getLogger(__name__)

        self.sampling = sampling_method

        self.check_sampling_method()

        self.load_each_matching_module(ref_dict, predict_dict, arg_dict)

        return None

    def check_sampling_method(self):

        # check if the sampling method has attributes:

        if (not hasattr(self.sampling, "run")):

            self.logger.error("The sampling_method object does "
                              "not have 'run' attributes")

            sys.exit("Check errors in the log file")

        if (not hasattr(self.sampling, "exit")):

            self.logger.error("The sampling_method object does "
                              "not have 'exit' attributes")

            sys.exit("Check errors in the log file")

        return None

    def input_keyword_to_module_keyword(self, input_keyword):

        keyword_dict = {
                        "force": "force_matching",
                        "rdf": "rdf_matching",
                        "isobar": "isobar_matching"
                        }

        module_name = keyword_dict.get(input_keyword)

        if (module_name is None):

            self.logger.error("The matching type %s not found;"
                              "Plase use: 'force', 'rdf', 'isobar' "
                              "or add the customized matching type"
                              % input_keyword)

            sys.exit("Check errors in the log file")

        else:

            return module_name

    def load_each_matching_module(self, ref_dict, predict_dict, arg_dict):

        self.load_objective_lst = []

        for every_type, _, _ in zip(ref_dict.keys(),
                                    predict_dict.keys(),
                                    arg_dict.keys()):

            module_name = self.input_keyword_to_module_keyword(every_type)

            import_path = ("objective" +
                           "."+"%s" % module_name +
                           "." +
                           "%s" % module_name)

            loaded_matching = importlib.import_module(import_path)

            self.check_loaded_module(loaded_matching)

            ref_sub_dict = ref_dict[every_type]

            predict_sub_dict = predict_dict[every_type]

            arg_sub_dict = arg_dict[every_type]

            for keys, _, _ in zip(ref_sub_dict.keys(),
                                  predict_sub_dict.keys(),
                                  arg_sub_dict.keys()):

                ref_address = ref_sub_dict[keys]

                predict_address = predict_sub_dict[keys]

                argument = arg_sub_dict[keys]

            initialize_objective = loaded_matching.load(ref_address,
                                                        predict_address,
                                                        argument,
                                                        self.output_folder)

            self.check_objective_attribute(initialize_objective)

            self.load_objective_lst.append(initialize_objective)

        return None

    def check_loaded_module(self, loaded_module):

        if (not hasattr(loaded_module, "load")):

            self.logger.error("The loaded matching module does "
                              "not have the 'load' attributes")

            sys.exit("Check errors in the log file")

        return None

    def check_objective_attribute(self, objective):

        if (not hasattr(objective, "optimize")):

            self.logger.error("The loaded objective function does "
                              "not have the 'optimize' attributes")

            sys.exit("Check errors in the log file")

        return None

    def optimize(self, fftype, force_field_parameters, status):

        # run sampling:

        self.sampling.run(fftype, force_field_parameters)

        # check sampling status:

        job_successful = self.sampling.exit()

        if (not job_successful):

            self.logger.error("Sampling method does not exit successfully !")

            sys.exit("Check errors in the log file")

        sum_objective = 0

        for job in self.load_objective_lst:

            sum_objective += job.optimize()

            # if the rename file attributes exist, rename the file needed

            if (hasattr(job, 'rename')):

                job.rename(status)

        return sum_objective

    def update(self, current_obj, best_obj, status):

        if (current_obj < best_obj):

            for job in self.load_objective_lst:

                if (hasattr(job, 'update')):

                    job.update(status)

        return None
