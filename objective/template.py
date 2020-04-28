

class load(): 

    def __init__(ref_address_tple,predict_address_tple,arg_string): 
    
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

force_1 = load(("/project/palmer/Jingxiang/referencedata/300K"),("/project/palmer/Jingxiang/predicted_data/300K"),"argument") 
force_2 = load(("/project/palmer/Jingxiang/referencedata/250K"),("/project/palmer/Jingxiang/referencedata/250K"),"argument") 
force_3 = load(("/project/palmer/Jingxiang/referencedata/200K"),("/project/palmer/Jingxiang/referencedata/200K"),"argument") 

force_1.optimize() 
force_2.optimize() 
force_3.optimize() 
