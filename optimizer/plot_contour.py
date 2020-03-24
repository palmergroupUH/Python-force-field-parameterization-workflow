import numpy as np 
import matplotlib.pyplot as plt 
import objective.test_optimizer

#def generate_

x_limit = np.arange(-8,8,0.15) 

y_limit = np.arange(-8,8,0.15) 

Himmelblau_test = objective.test_optimizer.Himmelblau() 

z_mat = np.zeros((x_limit.size,y_limit.size))

counter_x = 0 

for x in x_limit: 

    counter_y = 0 

    for y in y_limit: 

        z_mat[counter_x,counter_y] = np.log10(Himmelblau_test.evaluate(np.array([x,y])))  
        
        counter_y += 1 

    counter_x += 1 

for i in range(0,40,2):  

    vertices = np.loadtxt("vertices_%d.txt"%i)
    
    plt.scatter(vertices[:,0],vertices[:,1],s=1,color="r",alpha=1-0.02*i) 

    plt.plot(vertices[:4,0],vertices[:4,1],color="r",alpha=1-0.02*i) 

plt.contour(x_limit,y_limit,np.transpose(z_mat),20) 

plt.savefig("Himmelblau_plot.png",dpi=300)

plt.show()

