import numpy as np 


a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]]) 

print ( a.shape)

with open("gg.txt","w") as output:

    np.savetxt(output, np.c_[[ i for i in a ]]) 
