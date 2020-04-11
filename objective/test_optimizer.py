import numpy as np 
import matplotlib.pyplot as plt 

class Rosenbrock(): 

    def __init__(self,x_ranges,y_ranges):

        self.x_limit = np.arange(x_ranges[0],x_ranges[1],x_ranges[-1] ) 

        self.y_limit = np.arange(y_ranges[0],y_ranges[1],y_ranges[-1] ) 

        self.z_mat = np.zeros((self.x_limit.size,self.y_limit.size)) 

        counter_x = 0 
       
        for x in self.x_limit: 

            counter_y = 0 

            for y in self.y_limit: 

                self.z_mat[counter_x,counter_y] = np.log10(self.evaluate(np.array([x,y])))  
    
                counter_y += 1 

            counter_x += 1 

        return None 

    def visualize(self): 

        plt.xlabel("x") 

        plt.ylabel("y") 

        plt.contour(self.x_limit,self.y_limit,np.transpose(self.z_mat),20) 

        plt.show() 
        return None 

    def evaluate(self,indp_var):
            
        sum_f_x = 0   
     
        for i in range(0,indp_var.size-1): 
         
            sum_f_x = sum_f_x + (1 - indp_var[i] )**2 + 100*(indp_var[i+1]- indp_var[i]**2)**2  

        return sum_f_x 
    
    def optimize(self,para_type_lst,parameters):    

        predict_val = self.evaluate(parameters) 

        return np.sum((predict_val)**2)  


class Himmelblau():

    def __init__(self,x_ranges,y_ranges):

        self.x_limit = np.arange(x_ranges[0],x_ranges[1],x_ranges[-1] ) 

        self.y_limit = np.arange(y_ranges[0],y_ranges[1],y_ranges[-1] ) 

        self.z_mat = np.zeros((self.x_limit.size,self.y_limit.size)) 

        counter_x = 0 
       
        for x in self.x_limit: 

            counter_y = 0 

            for y in self.y_limit: 

                self.z_mat[counter_x,counter_y] = np.log10(self.evaluate(np.array([x,y])))  
    
                counter_y += 1 

            counter_x += 1 

        return None 

    def visualize(self): 

        plt.xlabel("x") 
        plt.ylabel("y") 
        plt.contour(self.x_limit,self.y_limit,np.transpose(self.z_mat),20) 
        plt.show() 

        return None 

    def evaluate(self,indp_var): 

        return (indp_var[0]**2 + indp_var[1] - 11 )**2 + (indp_var[0]+indp_var[1]**2 -7)**2  

    def optimize(self,para_type_lst,parameters): 

        predict_val = self.evaluate(parameters) 

        return np.sum((predict_val)**2)  


