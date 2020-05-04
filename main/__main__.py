#!/usr/bin/env python3
# Standard python library
import numpy as np
import IO.user_provided
import objective.setup_objective
import optimizer.gradient_free
import objective.compute_objective

# Local library:

# Third party library:


def main():

    #------------------------------------------------------------------------------
    #                       Taking the input from user                             
    #------------------------------------------------------------------------------
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
    # ( you don't have to do anything with it )

    # TOTAL_CORES: Number of cores assigned by slurm scheduler
    # INPUT: a string of given input file name
    # JOBID: a combination of Slurm job id and user-provided id

    main_logger, TOTAL_CORES, INPUT, JOBID,Ref,prep = (IO
                                              .user_provided
                                              .from_command_line()
                                              .finish_reading())

    #------------------------------------------------------------------------------
    #                           Set up the workflow                                
    #------------------------------------------------------------------------------
    # set up working folders and sampling methods ...
    ref_dict, predict_dict, argument_dict, LAMMPS, last_line = (objective
                                                                .setup_objective
                                                                .setup(
                                                                    INPUT,
                                                                    TOTAL_CORES,
                                                                    JOBID,
                                                                    overwrite=True,
                                                                    Ref_folder=Ref,
                                                                    prep_folder=prep)
                                                                .finish())

    #------------------------------------------------------------------------------
    #                           Initialize objective functions                     
    #------------------------------------------------------------------------------
    # instantiate the object of each matching type ...
    eval_objective = (objective
                      .compute_objective
                      .prepare(
                        ref_dict,
                        predict_dict,
                        argument_dict,
                        LAMMPS))

    # eval_objective: a Python list
    # each objective has attributes of "optimize"
    # [ objective1, objective2, objective3 ... ]

    #------------------------------------------------------------------------------
    #                           start optimization                                 
    #------------------------------------------------------------------------------

    # initialize optimizer ...
    optimize_fm = (optimizer
                   .gradient_free
                   .NelderMeadSimplex(
                       INPUT,
                       eval_objective,
                       skipped=last_line,
                       Output=JOBID+"/Output"))

    # run optimization ...
    optimize_fm.run_optimization()

    return None 

if ( __name__=="__main__"): 

    main() 
