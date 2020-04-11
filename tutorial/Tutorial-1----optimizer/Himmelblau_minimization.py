import optimizer.gradient_free
import objective.test_optimizer



x = [-8,8,0.15]  # start, end, step

y =[-8,8,0.15] # start, end, step

Himmelblau_test_obj = objective.test_optimizer.Himmelblau(x,y)  

#Himmelblau_test.visualize()

optimize_Himmelblau = optimizer.gradient_free.NelderMeadSimplex("in_himmelblau",Himmelblau_test_obj,skipped=0)

optimize_Himmelblau.run_optimization()

 


