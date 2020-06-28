# Python standard library:
import logging
import argparse
import numpy as np
import sys

# Local library:
import IO.check_type

# Third-party libraries:


class from_command_line():

    @classmethod
    def __init__(cls,
                 jobID=None,
                 total_cores=None,
                 input_file=None,
                 mode=None,
                 ref_address=None,
                 prep_address=None):

        if (mode is None):

            cls.MODE = "run"

        else:

            cls.MODE = mode

        if (jobID is not None):

            cls.JOBID = str(jobID)

            cls.logger = cls.Set_Run_Mode(cls.JOBID + ".log", cls.MODE)

        if (total_cores is not None):

            cls.TOTAL_CORES = total_cores

        if (input_file is not None):

            cls.INPUT = input_file

        if (ref_address is not None):

            cls.Ref_data = ref_address

        else:

            cls.Ref_data = "../ReferenceData"

        if (prep_address is not None):

            cls.prep_data = prep_address

        else:

            cls.prep_data = "../prepsystem"

        all_options = np.array([total_cores, jobID, input_file])

        # if None of total_cores ,jobID ,
        # input_file assigned, then use the command line options

        if (np.all(all_options == None)):

            cls.Take_Command_Line_Args()

            cls.set_global()

        if (np.any(all_options is not None) and np.any(all_options is None)):

            sys.exit("ERROR: either assign all values for arguments "
                     "in the class constructors "
                     "or read all input from command line")

        # check the following mandatory attributes

        cls.check_necessary_attributes("JOBID")
        cls.check_necessary_attributes("TOTAL_CORES")
        cls.check_necessary_attributes("INPUT")
        cls.check_necessary_attributes("logger")
        cls.check_necessary_attributes("Ref_data")
        cls.check_necessary_attributes("prep_data")

        # check the type of user-provided input:

        cls.check_total_cores()

        return None

    @classmethod
    def finish_reading(cls):

        return (cls.logger,
                cls.TOTAL_CORES,
                cls.INPUT,
                cls.JOBID,
                cls.Ref_data,
                cls.prep_data)

    @classmethod
    def check_necessary_attributes(cls, attribute):

        if (not hasattr(cls, attribute)):

            sys.exit('global variable: "%s" not found in either '
                     'command line or passed argument' % attribute)

        return None

    @classmethod
    def check_total_cores(cls):

        if (not IO.check_type.is_int(cls.TOTAL_CORES)):

            cls.logger.error("ERROR: varable: 'total_cores' "
                             "must be an integer !")

            sys.exit("Check errors in log file ! ")

        return None

    @classmethod
    def Take_Command_Line_Args(cls):

        parser = argparse.ArgumentParser(
                 description=("This is a Python software package "
                              "implementing a force-field "
                              "parameters optimization workflow"))

        parser.add_argument("-c",
                            "--cores",
                            type=int,
                            required=True,
                            help="Number of cores requested")

        parser.add_argument("-i",
                            "--input",
                            type=str,
                            required=True,
                            help="input file name")

        parser.add_argument("-j",
                            "--job",
                            type=str,
                            required=True,
                            help=("Specify a job ID that will be "
                                  "attached to a job folder and log file"))

        parser.add_argument("-m",
                            "--mode",
                            type=str,
                            required=False,
                            default="run",
                            help=("(Optional) Choose 'run' "
                                  "or 'debug'. Default is 'run' "))

        parser.add_argument("-Ref",
                            "--ReferenceData",
                            type=str,
                            required=False,
                            default="../ReferenceData",
                            help=("(Optional) Provide the path to "
                                  "Reference data folder. Default path is "
                                  "'../ReferenceData'"))

        parser.add_argument("-prep",
                            "--prepsystem",
                            type=str,
                            required=False,
                            default="../prepsystem",
                            help=("(Optional) Provide the path to "
                                  "prepsystem folder. Default path is "
                                  "'../prepsystem'"))

        args = parser.parse_args()

        cls.argument = dict(args.__dict__.items())

        return None

    @classmethod
    def set_global(cls):

        cls.JOBID = cls.argument["job"]

        cls.TOTAL_CORES = cls.argument["cores"]

        cls.INPUT = cls.argument["input"]

        cls.logger = cls.Set_Run_Mode(cls.JOBID + ".log", cls.MODE)

        cls.Ref_data = cls.argument["ReferenceData"]

        cls.prep_data = cls.argument["prepsystem"]

        return None

    @classmethod
    def Select_Run_Mode(cls, arg):

        mode = {
                "debug": logging.DEBUG,
                "run": logging.INFO
                }

        return mode[arg]

    @classmethod
    def Select_Formatter(cls, arg):

        mode = {
                "debug": "%(asctime)s - %(name)s - \
                         %(levelname)s - %(message)s",
                "run": "%(message)s"
               }

        return mode[arg]

    @classmethod
    def Set_Run_Mode(cls, logname, mode):

        logger = logging.getLogger()

        logger.setLevel(cls.Select_Run_Mode(mode))

        fh = logging.FileHandler(logname, mode="w")

        formatter = logging.Formatter(cls.Select_Formatter(mode))

        fh.setFormatter(formatter)

        logger.addHandler(fh)

        return logger


# module methods:
def keyword_exists(argument_str, keyword):

    try:

        keyword_indx = argument_str.index(keyword)

        return keyword_indx

    except ValueError:

        # '-1' means no such keyword exists in this string
        return -1
