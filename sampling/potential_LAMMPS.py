##############################################################################

# Python-force-field-parameterization-workflow:
# A Python Library for performing force-field optimization

#

# Authors: Jingxiang Guo, Jeremy Palmer

#

# Python-force-field-parameterization-workflow is free software;
# you can redistribute it and/or modify it under the terms of the
# MIT License 

# You should have received a copy of the MIT License along with the package.

##############################################################################


"""
This module contains a class to invoke LAMMPS
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

# Standard library:
import logging
import os
import itertools
import sys

# Local library:

# Third-parties library:
# GMSO
import mbuild as mb
import gmso
from gmso.external.convert_mbuild import from_mbuild
from gmso.formats.top import write_top
from gmso.formats import write_lammpsdata
from unyt import unyt_quantity 


# This module defines the force field output format for LAMMPS
# Customized pair_style function returns:
# A dictionary with the filename as "key", and its content as "values"

def choose_lammps_potential(ptype, force_field_parameters):

    potential_logger = logging.getLogger(__name__)

    potential_logger.debug("function:choose_lammps_potential "
                           "entered successfully")

    # a list of available LAMMPS potential functional:

    potential_type = {
                      "tersoff": __pair_style_tersoff,
                      "tersoff/table": __pair_style_tersoff,
                      "stillinger_weber": __pair_style_sw,
                      "lj/cut": __pair_style_lj_cut,
                      "buck/coul/long": __pair_style_buck_coul_long, 
                      "lj/smooth/linear": __pair_style_lj_smooth_linear_GMSO
                      }

    # raise the errors and exit the program if
    # requested potential is not defined

    if (ptype not in potential_type.keys()):

        potential_logger.error(
                               "ERROR: LAMMPS potential type: "
                               " %s  is invalid: \n" % ptype +
                               "Solutions: \n" +
                               "1. Check the spelling\n" +
                               "2. Define a customized force field "
                               "named %s in potential.py\n" % ptype +
                               "Currently available potential types are: " +
                               " , ".join(pt for pt in potential_type.keys()) +
                               "\n")

        sys.exit("Error messages found in the log file")

    # choose the chosen output force field

    chosen_potential = potential_type[ptype]

    output_force_field_dict = chosen_potential(ptype, force_field_parameters)

    potential_logger.debug("function:choose_lammps_potential "
                           "returned successfully; Potential type: "
                           "%s is used ..." % ptype)

    return output_force_field_dict


def propagate_force_field(wk_folder_tple, output_force_field_dict):

    potential_logger = logging.getLogger(__name__)

    potential_logger.debug("function: propagate_force_field "
                           "entered successfully !")

    for every_type in wk_folder_tple:

        for each_folder in every_type:

            for output_file in output_force_field_dict:

                output_content = output_force_field_dict[output_file]
                if (len(output_content) > 1 
                    and output_content[0] == "TurnOnGMSO"):
                    pass 
                    #write_lammpsdata = output_content[1] 
                    #filename = os.path.join(each_folder, output_file)
                    #write_lammpsdata(output_content[2], filename, output_content[-1]) 

                else: 
                    filename = os.path.join(each_folder, output_file)

                    with open(filename, "w") as output:

                        for line in output_content:

                            output.write(line)

    potential_logger.debug("function:propagate_force_field "
                           " returned successfully; force-field parameters ")

    return None


def __pair_style_lj_smooth_linear_GMSO(ptype, force_field_parameters):

    # output dictionary:
    # Generate a small box of Argon atoms using mBuild

    # output dictionary :
    force_field_dict = {}

    ar = mb.Compound(name='Ar')

    # (1.3954 g/cm^3 / 39.948 amu) * (3 nm) ^3
    packed_system = mb.fill_box(
        compound=ar,
        n_compounds=512,
        box=mb.Box([4.22187, 4.22187, 4.22187]),
    )

    # Convert system to a backend object
    top = from_mbuild(packed_system)
    lamp_data_name = "ar.lmp"
    force_field_dict[lamp_data_name] = ("TurnOnGMSO", write_lammpsdata, top, "atomic") 
     
    return force_field_dict 

def __pair_style_lj_cut(ptype, force_field_parameters):

    # output dictionary :

    force_field_dict = {}

    # define the filename

    include_file = "force_field_parameters"

    # define the command for each filename

    lammps_cmd_comment = "#pair style: %s is used \n" % ptype

    lammps_cmd_1 = "pair_style   lj/cut %.3f" % force_field_parameters[0]

    lammps_cmd_2 = "pair_coeff * * %.9f %.9f" % (force_field_parameters[1],
                                                 force_field_parameters[2])

    lammps_cmd_3 = "pair_modify tail yes"

    force_field_dict[include_file] = (lammps_cmd_comment,
                                      lammps_cmd_1,
                                      lammps_cmd_2,
                                      lammps_cmd_3)

    return force_field_dict


def __pair_style_sw(ptype, force_field_parameters):

    # output dictionary :

    force_field_dict = {}

    # define the filename

    potential_file = "mW.sw"

    # define the filename

    include_file = "force_field_parameters"

    lammps_cmd_comment = "#pair style: %s is used \n" % ptype

    element = "WT"

    command1 = "pair_style   sw\n"

    command2 = ("pair_coeff" + " " + "* *" +
                " " + potential_file + " " + element + "\n")

    pair_command = ((element+" ")*3 +
                    " ".join(str(para) for para in force_field_parameters))

    force_field_dict[include_file] = (lammps_cmd_comment,
                                      command1,
                                      command2)

    force_field_dict[potential_file] = (pair_command)

    return force_field_dict


def __pair_style_tersoff(ptype, force_field_parameters):

    # output dictionary:

    force_field_dict = {}

    # define the filename

    potential_file = "WT_ML-BOP.tersoff"

    # define the filename

    include_file = "force_field_parameters"

    lammps_cmd_comment = "# pair style: %s is used \n" % ptype

    element = "WT"

    if ("table" in ptype):

        command1 = "pair_style   tersoff/table\n"

    else:

        command1 = "pair_style   tersoff\n"

    command2 = "pair_coeff" + " * * " + potential_file + " " + element

    pair_command = ((element + " ")*3 +
                    " ".join(str(para) for para in force_field_parameters))

    force_field_dict[include_file] = (lammps_cmd_comment, command1, command2)

    force_field_dict[potential_file] = (pair_command)

    return force_field_dict


def __pair_style_buck_coul_long(ptype, force_field_parameters):

    # output dictionary:

    force_field_dict = {}

    # define the filename

    include_file = "force_field_parameters"

    # comment of included potential file

    lammps_cmd_comment = "#pair style: %s is used \n" % ptype

    # lammps command:

    lammps_command_1 = ("pair_style buck/coul/long %.3f"
                        % force_field_parameters[0])

    lammps_command_2 = ("pair_coeff    1 1 %.5f %.5f %.5f"
                        % (force_field_parameters[1],
                           force_field_parameters[2],
                           force_field_parameters[3]))

    lammps_command_3 = ("pair_coeff    2 2 %.5f %.5f %.5f"
                        % (force_field_parameters[4],
                           force_field_parameters[5],
                           force_field_parameters[6]))

    lammps_command_4 = ("pair_coeff    1 2 %.5f %.5f %.5f"
                        % (force_field_parameters[7],
                           force_field_parameters[8],
                           force_field_parameters[9]))

    lammps_command_5 = "kspace_style pppm 1.0e-4"

    force_field_dict[include_file] = (lammps_cmd_comment,
                                      lammps_command_1,
                                      lammps_command_2,
                                      lammps_command_3,
                                      lammps_command_4,
                                      lammps_command_5)

    return force_field_dict
