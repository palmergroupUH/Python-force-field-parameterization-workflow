# Python standard library: 
import numpy as np 

# Local library: 
import IO.reader 
# fortran API: 
import computerdf

# Third-party libraries: 
import MDAnalysis 

class load(): 

    # use class to define objective functions

    # 1. instantiate many objective functions 

    # 2. hide implementation details: variables, functions .... 

    # 3. generic input and output 

    def __init__(ref_address_tple,predict_address_tple,arg_string): 

        self.compute_norm() 

        self.parse_argument() 

        self.load_data() 
    
        return None 

    def define_filename(self): 

        self.ref_file = "Ref.properties"
   
        self.predict_file = "predict.properties" 

        return None 

    def parse_argument(self):

        return None 

    def compute_norm(self): 
    
        self.norm = load_data(ref) 

        return None 

    def check_data(self):  

        return None     

    def load_data(self): 

        return None 

    def compute_objective(self): 

        ref_data = self.load_data(ref) 

        predict_data = self.load_data(predict) 

        for i_ref,i_predict in zip(ref_data,predict_data): 

            self.objective_lst.append( (predict_data[i_predict]-ref_data[i_ref] )**2/self.norm )  

        return None 

    def optimize(self): 

        sum_obj = 0 

        for i in self.objective_lst:         
   
            sum_obj += self.objective_lst[i]  
    
        return sum_obj   

# initialize an object using "load" class 
force_1 = load(("/project/palmer/Jingxiang/referencedata/300K"),("/project/palmer/Jingxiang/predicted_data/300K"),"argument") 
force_2 = load(("/project/palmer/Jingxiang/referencedata/250K"),("/project/palmer/Jingxiang/referencedata/250K"),"argument") 
force_3 = load(("/project/palmer/Jingxiang/referencedata/200K"),("/project/palmer/Jingxiang/referencedata/200K"),"argument") 

# 
obj = force_1.optimize() 
obj = force_2.optimize() 
obj = force_3.optimize() 
