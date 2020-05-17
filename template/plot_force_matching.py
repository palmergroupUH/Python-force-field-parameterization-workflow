# Python standard library: 
import numpy as np 
import matplotlib.pyplot as plt 

# Local library: 
import IO.reader 

# Third-party libraries: 

# Energy matching
# read the energy data: 

# guess: 
# provide eng and force file address:  
# Energy data: 

guess_file = "force_mathcing_387865/Output/mW_300K_1bar_500_guess.eng"
ref_file = "/project/palmer/Jingxiang/ours_optimization/tutorial/force_matching_tutorial/ReferenceData/force/mW_300K_1bar_500/Ref.eng" 
best_predict = "force_mathcing_387865/Output/mW_300K_1bar_500_best.eng"

# Force data: 
ref_force_data_address = "/project/palmer/Jingxiang/ours_optimization/tutorial/force_matching_tutorial/ReferenceData/force/mW_300K_1bar_500/Ref.force"
guess_force_data_address = "force_mathcing_387865/Output/mW_300K_1bar_500_guess.force"
best_force_data_address = "force_mathcing_387865/Output/mW_300K_1bar_500_best.force"


num_lines,num_columns = IO.reader.get_lines_columns(guess_file) 

guess_eng = IO.reader.loadtxt(guess_file,num_lines,skiprows=1,return_numpy=True)

# reference:

num_lines,num_columns = IO.reader.get_lines_columns(ref_file)

ref_eng = IO.reader.loadtxt(ref_file,num_lines,skiprows=0,return_numpy=True) 

# best: 

num_lines,num_columns = IO.reader.get_lines_columns(best_predict)

best_eng = IO.reader.loadtxt(best_predict,num_lines,skiprows=1,return_numpy=True)

bins = 50

plt.hist(guess_eng,bins,density=True,label="guess",color="b")
plt.hist(ref_eng,bins,density=True,label="Ref",color="k")
plt.hist(best_eng,bins,density=True,alpha=0.6,label="best_predicted",color="r")

plt.xlabel("Pe")

plt.ylabel("P(Pe)")

plt.legend(loc="upper right")
plt.savefig("energy_distribution.png")


# Force matching: 
# read the force data in parallel: 

start_at = 1 # The nth configuration to start with ( by default starts with 1st configuration)

work_load = 500 # total number of configurations

num_cores = 1 # total number of cores assigned 

buffer_size = 2000 # total number of configuration read into memory at once for each time 

total_atoms = 512

work_flow = IO.reader.parallel_assignment(start_at,work_load,num_cores,buffer_size) 
# work_flow: A python list containing
# [((start nconfigs),(start,nconfigs)),((start,nconfigs) ...] 

ref_output_lst = IO.reader.read_LAMMPS_traj_in_parallel(ref_force_data_address,num_cores,total_atoms,work_load,first=1,buffer_size=buffer_size)

guess_output_lst = IO.reader.read_LAMMPS_traj_in_parallel(guess_force_data_address,num_cores,total_atoms,work_load,first=1,buffer_size=buffer_size)

best_output_lst = IO.reader.read_LAMMPS_traj_in_parallel(best_force_data_address,num_cores,total_atoms,work_load,first=1,buffer_size=buffer_size)

sum_sqr = 0

# plot Ref vs guess:  
# new figure 
plt.figure() 

x_force = np.arange(-100,100) 
y_force = x_force 

for ref_output,guess_output in zip(ref_output_lst,guess_output_lst):
    
     ref_data = ref_output.get()

     guess_data = guess_output.get() 

     plt.scatter(ref_data,guess_data)       

plt.xlim([-30,40])
plt.ylim([-30,40])

plt.xlabel("reference forces: fx,fy,fz (kcal/mol)")    
plt.ylabel("guess forces: fx,fy,fz(kcal/mol)")    
plt.plot(x_force,y_force,label="forces are equal")
plt.legend(loc="upper center")
plt.savefig("guess_force.png")

plt.figure() 

for ref_output,best_output in zip(ref_output_lst,best_output_lst):
    
     ref_data = ref_output.get()

     best_data = best_output.get() 

     plt.scatter(ref_data,best_data)       

plt.xlim([-30,40])
plt.ylim([-30,40])
plt.plot(x_force,y_force,label="forces are equal")
plt.xlabel("reference forces: fx,fy,fz (kcal/mol)")    
plt.ylabel("best forces: fx,fy,fz (kcal/mol)")    
plt.legend(loc="upper center")
plt.savefig("best_force.png")


