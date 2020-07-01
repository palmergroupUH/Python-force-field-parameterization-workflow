SHELL = /bin/sh

# fortran compiler 
FC = ifort #

FFLAGS_D = -fPIC -shared

FFLAGS_S = -c 

INC_DIR = .include/

LIB_DIR = lib/

link_to_system = -L../general/lib/ 

libfile = -lforsys

sys_include = -I ../general/include

local_include = -module include/

system_file= ../general/src/system_mod.f90

constants_file= ../general/src/constants_mod.f90

all: lib/libpair_correlation.so 

# compile all source codes into dynamic library 
#
$(LIB_DIR)libpair_correlation.so: src/compute_rdf_mod.f90

	$(FC) $(FFLAGS_D) $(system_file) $(constants_file) src/compute_rdf_mod.f90  -o $(LIB_DIR)libpair_correlation.so -module include/ 
