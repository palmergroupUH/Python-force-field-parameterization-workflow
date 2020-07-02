SHELL = /bin/sh

# fortran compiler 
FC = ifort #

FFLAGS_D = -fPIC -shared

FFLAGS_S = -c -fPIC

INC_DIR = include/

LIB_DIR = lib/

include_folder = -I include/ 

all: $(LIB_DIR)libforsys.a  

system_mod.o: src/system_mod.f90 

	@ mkdir -p $(INC_DIR)

	$(FC) $(FFLAGS_S) src/system_mod.f90  

	@ mv *.mod $(INC_DIR)

constants_mod.o: src/constants_mod.f90

	@ mkdir -p $(INC_DIR)

	$(FC) $(FFLAGS_S) src/constants_mod.f90 $(include_folder)

	@ mv *.mod $(INC_DIR)

$(LIB_DIR)libforsys.a: system_mod.o constants_mod.o  

	@ mkdir -p $(LIB_DIR)

	ar -cvq $(LIB_DIR)libforsys.a system_mod.o constants_mod.o 

.PHONY: all clean count

clean:

	rm *.o 

