import mbuild as mb 

# initialize a compound object
cg_water = mb.Compound()


# create a coarse-grained water bead,
# "_xx" name used for a coarse-grained system 
cg_bead = mb.Particle(name='_H2O', pos=[0, 0, 0])

# add each bead to 
cg_water.add([cg_bead])

# fill a box with the defined "compound" 
# box size is chosen to match with density 0.997g/cm3 for water 
cg_water_box = mb.fill_box(compound=cg_water,
                           n_compounds=512,
                           box=[2.4859,2.4859,2.4859],
                           seed=2020)

# use a .xml file to change the molar mass of coarse-grained bead to water's molar mass
# save the configuration in LAMMPS data file format
cg_water_box.save('mW.lmp',
                  forcefield_files="water.xml",
                  overwrite=True,
                  foyer_kwargs={"assert_bond_params":False})
                                

