import gradient_free 
import objective.test_optimizer 
import matplotlib.pyplot as plt 

# test Rosenbrock function: 1 global minimum ( 1,1 ) 
"""

Rosenbrock_obj = objective.test_optimizer.RosenbrockFunction()  

optimize_Rosenbrock = gradient_free.NelderMeadSimplex("in_rosenbrock",Rosenbrock_obj,skipped=0)

optimize_Rosenbrock.run_optimization() 

"""

# test Himmelblau function: 4 local minimum 
# minima 1: x= 3.0,        y=2.0 
# minima 2: x= -2.805118,  y=3.131312 
# minima 3: x= -3.779310,  y=-3.283186
# minima 4: x= 3.584428,   y= -1.848126 

Himmelblau_obj = objective.test_optimizer.Himmelblau() 

optimize_Himmelblau = gradient_free.NelderMeadSimplex("in_himmelblau",Himmelblau_obj,skipped=0)

optimize_Himmelblau.run_optimization() 


