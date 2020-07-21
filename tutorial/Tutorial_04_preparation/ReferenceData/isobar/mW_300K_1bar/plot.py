import numpy as np 
import os 
import matplotlib.pyplot as plt 

T  = [230,240,250,260,270,280]

for temp in T: 

    data_file = os.path.join(str(temp),"Ref.density")

    dens_data = np.average(np.loadtxt(data_file)) 

    print(dens_data) 
    plt.scatter(temp,dens_data)

plt.ylim([0.995,1.01]) 
plt.xlabel("Temperature") 
plt.ylabel("Density (g/cm3)") 

plt.show() 
    

 
 


