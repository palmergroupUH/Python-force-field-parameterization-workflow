# Standard library:
import logging
import os 
import itertools 
import sys 

# This module defines the force field output format for LAMMPS 
# Customized pair_style function returns a dictionary with the filename as "key", and its content as "values"  

def choose_lammps_potential(ptype,force_field_parameters):

	potential_logger = logging.getLogger(__name__) 
	
	potential_logger.debug("function:choose_lammps_potential entered successfully"  ) 

	# a list of available LAMMPS potential functional:  

	potential_type = { 

		"tersoff": __pair_style_tersoff,
		"tersoff/table": __pair_style_tersoff,
		"stillinger_weber": __pair_style_sw,
		"lj/cut": __pair_style_lj_cut,
		"buck/coul/long": __pair_style_buck_coul_long  
		} 

	# raise the errors and exit the program if requested potential is not defined

	if ( not ptype in potential_type.keys() ):
	
		potential_logger.error( "ERROR: LAMMPS potential type %s  is invalid: \n"%ptype +
						    "Solutions: \n" + 
							"1. Check the spelling\n" + 
							"2. Define a customized force field named %s in potential.py\n"%ptype +
							"Currently available potential types are: "+ " , ".join(pt for pt in potential_type.keys()) +"\n") 

		sys.exit("Error messages found in the log file") 
	
	# choose the chosen output force field 

	chosen_potential = potential_type[ptype]

	output_force_field_dict = chosen_potential(ptype,force_field_parameters) 

	potential_logger.debug("function:choose_lammps_potential returned successfully ; Potential type:  %s is used ..."%ptype ) 
	
	return output_force_field_dict 

def propagate_force_field(wk_folder_lst,output_force_field_dict):

	potential_logger = logging.getLogger(__name__)

	potential_logger.debug("function:propagate_force_field entered successfully !")

	all_wk_folder_tp = tuple(itertools.chain.from_iterable( wk_folder_lst)) 

	for each_folder in all_wk_folder_tp: 

		for output_file in output_force_field_dict: 

			output_content = output_force_field_dict[output_file] 

			filename = os.path.join(each_folder,output_file)
			
			with open(filename,"w") as output: 

				for line in output_content: 

					output.write(line)  	

					#output.write("\n") 

	potential_logger.debug("function:propagate_force_field returned successfully ; force-field parameters ")

	return None  

def __pair_style_lj_cut(ptype,force_field_parameters): 

	# output dictionary : 

	force_field_dict = {} 

	# define the filename

	include_file = "force_field_parameters" 

	# define the command for each filename 

	lammps_cmd_comment ="#pair style: %s is used \n"%ptype 

	lammps_cmd_1 = "pair_style   lj/cut %.3f"%force_field_parameters[0] 

	lammps_cmd_2 = "pair_coeff * * %.9f %.9f"%(force_field_parameters[1],force_field_parameters[2]) 
	
	lammps_cmd_3 = "pair_modify tail yes"

	force_field_dict[include_file] =  (lammps_cmd_comment,lammps_cmd_1,lammps_cmd_2,lammps_cmd_3)

	return force_field_dict 

def __pair_style_sw(ptype,force_field_parameters):  

	# output dictionary : 

	force_field_dict = {}

	# define the filename 

	potential_file = "mW.sw" 

	# define the filename 

	include_file = "force_field_parameters"

	lammps_cmd_comment ="#pair style: %s is used \n"%ptype 

	element = "WT"

	command1 = "pair_style   sw\n" 	

	command2 = "pair_coeff" +  " " + "* *"  + " " + potential_file + " " + element + "\n"
	
	pair_command =  (element+" ")*3 + " ".join(str(para) for para in force_field_parameters)

	force_field_dict[include_file] = (lammps_cmd_comment,command1,command2 )

	force_field_dict[potential_file] = (pair_command)

	return force_field_dict	

def __pair_style_tersoff(ptype,force_field_parameters): 

	# output dictionary :  
		
	force_field_dict = {}

	# define the filename 

	potential_file = "WT_ML-BOP.tersoff" 

	# define the filename 

	include_file = "force_field_parameters"

	lammps_cmd_comment ="#pair style: %s is used \n"%ptype 

	if ( "table" in ptype): 

		command1 = "pair_style   tersoff/table" 	

	else: 

		command1 = "pair_style   tersoff"

	command2 = "pair_coeff" + " * * " + potential_filename + " " + element

	pair_command =  ( element + " " )*3 + " ".join(str(para) for para in force_field_parameters)

	force_field_dict[include_file] = (lammps_cmd_comment,command1,command2 ) 

	force_field_dict[potential_file] = (pair_command)

	return force_field_dict 	

def __pair_style_buck_coul_long(ptype,force_field_parameters):

	# output dictionary : 	

	force_field_dict = {} 

	# define the filename 

	include_file = "force_field_parameters"

	# comment of included potential file 

	lammps_cmd_comment ="#pair style: %s is used \n"%ptype

	# lammps command:  

	lammps_command_1 = "pair_style buck/coul/long %.3f"%force_field_parameters[0]

	lammps_command_2 = "pair_coeff      1 1 %.5f %.5f %.5f"%(force_field_parameters[1],force_field_parameters[2],force_field_parameters[3])

	lammps_command_3 = "pair_coeff      2 2 %.5f %.5f %.5f"%(force_field_parameters[4],force_field_parameters[5],force_field_parameters[6])

	lammps_command_4 = "pair_coeff      1 2 %.5f %.5f %.5f"%(force_field_parameters[7],force_field_parameters[8],force_field_parameters[9])

	lammps_command_5 = "kspace_style pppm 1.0e-4"

	force_field_dict[include_file] = (lammps_cmd_comment,lammps_command_1,lammps_command_2,lammps_command_3,lammps_command_4,lammps_command_5 )

	return force_field_dict 
