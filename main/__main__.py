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
This module contains the main function "optimize_main" to perform
the force-field optimization.

The executable "optimize" is invoked from the command-line interface. It
will first call "main()", which then call the function "optimize_main".
Some other command-line programs related to this package can be developed,
and invoked in an anaglous fashion. The example "clearJobID" is defined, and
used just like "optimize"


The "optimize_main" is composed of several instatiation of classes imported
from different modules, which are laid out in procedure-oriented fashion
so that the user can easily understand the whole workflow.
This should make the subsequent customization more transparant.

"""

#!/usr/bin/env python3
# Standard python library
import sys

# Local library:
import IO.user_provided
import objective.setup_objective
import optimizer.gradient_free
import objective.compute_objective

# Third party library:


def optimize_main():

    # force stdin,stderr,stdout to be unbuffered
    sys.stdout.flush()

    # -------------------------------------------------------------------------
    #                       Taking the input from user
    # -------------------------------------------------------------------------
    # This is the main program
    # Uncomment the following docstring if running the program interactively
    """
    main_logger,TOTAL_CORES,INPUT,JOBID,Ref,prep = (IO
                                                 .user_provided
                                                 .from_command_line(
                                                    jobID="1234",
                                                    total_cores=4,
                                                    input_file="in_obj")
                                                 .finish_reading())
    """

    # global variables:
    # main_logger: an object that defines the log file output
    # (you don't have to do anything with it)

    # TOTAL_CORES: The number of cores assigned by slurm scheduler
    # INPUT: The string of given input file name
    # JOBID: The combination of Slurm job id and user-provided id
    (main_logger,
     TOTAL_CORES,
     INPUT,
     JOBID,
     Ref,
     prep) = IO.user_provided.from_command_line().finish_reading()

    # -------------------------------------------------------------------------
    #                           Set up the workflow
    # -------------------------------------------------------------------------
    # set up working folders and sampling methods ...

    (ref_dict,
     predict_dict,
     argument_dict,
     LAMMPS,
     last_line) = objective.setup_objective.setup(INPUT,
                                                  TOTAL_CORES,
                                                  JOBID,
                                                  overwrite=True,
                                                  Ref_folder=Ref,
                                                  prep_folder=prep).finish()

    # -------------------------------------------------------------------------
    #                           Initialize objective functions
    # -------------------------------------------------------------------------
    # instantiate the object of each matching type ...

    eval_objective = objective.compute_objective.prepare(
                                                         JOBID,
                                                         ref_dict,
                                                         predict_dict,
                                                         argument_dict,
                                                         LAMMPS)

    # eval_objective: a Python list containing an instance of the objective
    # function class: "load" 
    # each instance has an attribute of "optimize"
    # [ objective1, objective2, objective3 ... ]

    # -------------------------------------------------------------------------
    #                           start optimization
    # -------------------------------------------------------------------------

    # initialize optimizer ...
    optimize_fm = optimizer.gradient_free.NelderMeadSimplex(INPUT,
                                                            eval_objective,
                                                            skipped=last_line,
                                                            output=JOBID)

    # run optimization ...
    optimize_fm.run_optimization()

    return None


# not implemented yet
def clear_job_id_main():

    return None


# main programs to be executed from the command-line interface
def main():

    # currently implemented
    if (sys.argv[0].find("optimize") >= 0):

        optimize_main()

    # not implemented yet
    elif (sys.argv[0].find("clearJobID") >= 0):

        clear_job_id_main()

    return None


if (__name__ == "__main__"):

    # execute the program from the command-line interface

    main()
