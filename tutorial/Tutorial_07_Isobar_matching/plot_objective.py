import numpy as np
import os 
import matplotlib.pyplot as plt 

folder="isobars_441698"

objective = "Output/best_objective.txt"

dump_freq = 5

best_objective = np.loadtxt(os.path.join(folder,objective))

iteration = np.arange(1,best_objective.size+1) 

plt.semilogy(iteration*dump_freq, best_objective)

plt.xlabel("Iteration")

plt.ylabel("Objective function")

plt.savefig("isobar_objective.png")

plt.show()
