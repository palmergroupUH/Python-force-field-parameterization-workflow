import numpy as np 


density = np.loadtxt("Ref.density") 

index = np.argsort(density[:,1]) 

print index 
