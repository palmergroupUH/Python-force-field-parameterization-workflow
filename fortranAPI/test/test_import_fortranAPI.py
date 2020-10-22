# Python standard library: 
import numpy as np 
import os 
from ctypes import CDLL

# Local library: 
# fortran API: 
import fortranAPI.IO
import fortranAPI.pair_correlation

# Third-party libraries: 
import pytest 


def test_import_IO_reader():

    # get the dynamic library path from the fortranAPI IO module:
    fortranlib_address = os.path.join(os.path.dirname(fortranAPI.IO.__file__),
                                      "lib")

    # Load the dynamic library of dcd trajectory reader:
    dcd_lib = CDLL(os.path.join(fortranlib_address, "libdcd_reader.so"))

    # Load the dynamic library of txt file reader:
    txt_lib = CDLL(os.path.join(fortranlib_address, "libtxt_reader.so"))

    # Load the dynamic library of txt file reader:
    txt_lib = CDLL(os.path.join(fortranlib_address, "libxyz_reader.so"))

    return None

def test_import_pair_correlation():

    # get the dynamic library path from the fortranAPI IO module:
    fortranlib_address = os.path.join(os.path.dirname(fortranAPI.pair_correlation.__file__),
                                      "lib")


    pcf_lib = CDLL(os.path.join(fortranlib_address, "libpair_correlation.so"))
    
    return None

def test_import_general():


    return None 
