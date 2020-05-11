# Python standard library 
import numpy as np 
import multiprocessing as mp 
import os 
import sys 
import itertools 
import ctypes 
import logging 
from ctypes import CDLL, POINTER, c_int, c_double,c_char_p,c_long,c_float,byref 

# Local library: 
import IO
import pair_correlation 
# Third-party library: 

"""
class load():

    def __init__(ref_address_tple,predit_address_tple,argument_tple): 

       
    def loaded_filename(self):  

    

    def set_file_address_and_check_status() 

        return None 

    def check_predicted_data_status(self):

    #return None 

    def parse_input(self):

"""

dcdfile = "/project/palmer/Jingxiang/ours_optimization/tutorial/IO_tutorial/test.dcd" 
cores = 1 
cutoff = 10.0  
num_bins = 200
buffersize = 1000  

rdf_calc = pair_correlation.RadialDistribution(dcdfile,cores,cutoff,num_bins,buffersize)

rdf_calc.dump_gr("dump.gr")
rdf_calc.dump_r2hr("dump.hr")

