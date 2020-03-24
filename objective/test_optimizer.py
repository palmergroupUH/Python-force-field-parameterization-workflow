import numpy as np 

class Rosenbrock(): 

    def __init__(self):

        return None 

    def evaluate(self,indp_var):
            
        sum_f_x = 0   
     
        for i in range(0,indp_var.size-1): 
         
            sum_f_x = sum_f_x + (1 - indp_var[i] )**2 + 100*(indp_var[i+1]- indp_var[i]**2)**2  

        return sum_f_x 
    
    def optimize(self,parameters):    

        predict_val = self.evaluate(parameters) 

        return np.sum((predict_val)**2)  


class Himmelblau():

    def __init__(self):

        return None 

    def evaluate(self,indp_var): 

        return (indp_var[0]**2 + indp_var[1] - 11 )**2 + (indp_var[0]+indp_var[1]**2 -7)**2  

    def optimize(self,parameters): 

        predict_val = self.evaluate(parameters) 

        return np.sum((predict_val)**2)  


