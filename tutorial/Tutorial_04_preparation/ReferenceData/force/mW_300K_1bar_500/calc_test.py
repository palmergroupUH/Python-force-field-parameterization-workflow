import numpy as np

N = 512

n_dof = 3*N - 3  

mol = 6.02214076*10**23

kb = 1.38064852*10**(-23) 

T = np.loadtxt("Ref.temp")

ke = np.loadtxt("Ref.kinetic")

virial = np.loadtxt("Ref.virial")

pressure = np.loadtxt("Ref.pressure")

volume = np.loadtxt("Ref.volume")

# compare ke

ke_calc = (n_dof*T*kb*mol)

print (virial*volume*10**(-30)*101325*mol)

print ("kinetic energy: ", ke_calc)

print (np.var(virial))
print (np.var(pressure)) 

# compare T
#T_calc = (2*ke*N*1000*4.184)/(kb*n_dof*mol)

#print ("P from LAMMPS: ", pressure)

# compare Pressure
#P_calc = 1/(101325*3.0*volume*10**-30)*(2*ke*N*1000*4.184)/mol + virial

#print (P_calc)

