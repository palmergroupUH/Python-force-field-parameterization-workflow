import numpy as np 

a = {} 

for i in range(10):

    a[i] = i + 1   

b = np.array(list(a.keys())) 

print (a.keys()[-1]) 
