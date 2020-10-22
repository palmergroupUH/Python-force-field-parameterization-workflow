# Python standard library: 
import numpy as np 

# Local library (my library): 
from optimizer.optimizer_mod import set_optimizer  

# Third-party libraries: 
import pytest  

@pytest.fixture(scope="module") 
def load_optimizer_mod(): 

    # load the optimizer input file 
    input_file = "sample_input"

    # skip reading first 26 lines of sample_input 
    # generate a log file called: optimizer.log 
    # stop reading at certain line of sample_input 

    template = set_optimizer.__new__(set_optimizer)
    #set_optimizer(input_file,logname="optimizer.log",skipped=26,stop_after=None) 
    input_dict = {1: ['5', '5'],
                  8: ['stillinger_weber', 
                     '6.589', 
                     '2.0925', 
                     '1.87', 
                     '29.15', 
                     '1.02', 
                     '-0.303333333333', 
                     '7.049556277', 
                     '0.6822245584', 
                     '3.2', 
                     '0.4', 
                     '0.0'],
                  13: ['1', '1','1', '1', '1', '1','0', '1', '1', '1','0'],  
                  20: ['10', '0', '5', '2','0','3'], 
                  27: ['25000', '1e-14', '1e-14'],  
                  37: ['Nelder-Mead', 'perturb'], 
                  39: ['-0.4', '-0.4', '0.4', '0.4', '0.4', '-0.4', '0.4', '-0.4', '-0.4']}  

    template.input_data_dict = input_dict             
    
    template.keys_lst = list(input_dict.keys()) 

    template.values_lst = list(input_dict.values()) 

    template.pointer = 0 

    return template 

def test_log(load_optimizer_mod): 

    logname = "optimizer.log"

    load_optimizer_mod.add_logger(logname) 
    
    return None 

def test_optimizer_args(load_optimizer_mod): 

    load_optimizer_mod.parse_dumping_freq() 

    load_optimizer_mod.parse_guess()

    load_optimizer_mod.parse_fit_and_fixed()

    load_optimizer_mod.parse_bounds()

    load_optimizer_mod.parse_termination()

    load_optimizer_mod.parse_optimizer() 

    load_optimizer_mod.check_input_parameters() 

    return None  

def test_constrain(load_optimizer_mod): 

    ary = np.array([6.189,5.3925, 1.80,  23.15, 1.20,  -0.333333333333, 0.6022245584,  4.0, -0.3, 0.0])  

    # check the "constrain" attributes 

    assert hasattr(load_optimizer_mod,"constrain")  

    load_optimizer_mod.constrain(ary) 

    assert ary[1] == 3.0  

    assert ary[8] == 0.0   

    return None 

def test_regroup_with_fixed(load_optimizer_mod): 

    ary = np.array([6.189,5.3925, 1.80,  23.15, 1.20,  -0.333333333333, 0.6022245584,  4.0, -0.3 ])  

    all_para = load_optimizer_mod.group_fixed(ary) 

    assert all_para[-1] == 0.0 

    assert all_para[6] == 7.049556277 

    return None 





