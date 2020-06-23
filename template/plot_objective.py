import numpy as np
import os 
import matplotlib.pyplot as plt 

folder="isobars_427416"

objective = "Output/best_objective.txt"

best_objective = np.loadtxt(os.path.join(folder,objective))

iteration = np.arange(1,best_objective.size+1) 

plt.semilogy(iteration, best_objective)

plt.xlabel("Iteration")

plt.ylabel("Objective function")

plt.savefig("isobar_objective.png")

plt.show()
