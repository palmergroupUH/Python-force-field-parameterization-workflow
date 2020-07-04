## Tutorial 03: Solving for Himmelblau function

# After you successfully install the package and activate a conda environment
from optimizer.gradient_free import NelderMeadSimplex
import numpy as np
import matplotlib.pyplot as plt 


class Himmelblau():

    def __init__(self, x_ranges, y_ranges):

        self.x_limit = np.arange(x_ranges[0],x_ranges[1], x_ranges[-1])

        self.y_limit = np.arange(y_ranges[0],y_ranges[1], y_ranges[-1])

        self.z_mat = np.zeros((self.x_limit.size, self.y_limit.size))

        counter_x = 0

        for x in self.x_limit:

            counter_y = 0

            for y in self.y_limit:

                self.z_mat[counter_x, counter_y] = np.log10(self.compute_z(np.array([x, y])))
    
                counter_y += 1

            counter_x += 1

        return None

    def visualize(self):

        plt.xlabel("x")
        plt.ylabel("y")
        plt.contour(self.x_limit, self.y_limit, np.transpose(self.z_mat), 20)
        plt.show()

        return None

    def compute_z(self, parameters):

        predict_val = ((parameters[0]**2 + parameters[1] - 11 )**2 +
                       (parameters[0] + parameters[1]**2 -7)**2)
        
        return predict_val

    # "update" must be here. For force-matching, rdf-matching ..., this function
    # will be used to update the best predicted properties. 
    def update(self, func_expand, best_func, status=None):

        pass

        return None 

    # method "optimize" must be here. the optimizer will assume every 
    # passed objective function will have a attribute of "optimize"
    # "para_type_lst", and "status" also must be here, though they are not used
    def optimize(self, para_type_lst, parameters, status=None):

        return self.compute_z(parameters) 

# input file name
input_file = "in_himmelblau"

# No lines skipped:
skipped_lines = 0

# The solution space of Himmelblau function
x = [-8, 8, 0.15]
y = [-8, 8, 0.15]

# initialize test objective functions
himmeblau_obj = Himmelblau(x, y)

# Visualize the solution space of Himmelblau
# local minmums:
# Solution 1: x = 3.0, y = 2.0 
# Solution 2: x = -2.8051, y = 3.1313
# Solution 3: x = -3.7793, y = -3.2832
# Solution 4: x = 3.5844, y = -1.8481
# local maximum: 
# Solution 1: x = -0.270845, y = -0.923039
himmeblau_obj.visualize() 

# initialize optimizer ...
optimize_himme = NelderMeadSimplex(input_file,
                                   himmeblau_obj,
                                   skipped=skipped_lines)
# Optimization starts ...
optimize_himme.run_optimization()

# Load the solutions:
with open("best_parameters.txt") as content: 

    for line in content:

        solution = np.array(line.split()).astype(np.float64)

    x, y = solution
    print("The Minimum found is x = %.4f, y = %.4f" % (x, y))

# initialize optimizer ...
optimize_himme = NelderMeadSimplex(input_file,
                                   himmeblau_obj,
                                   skipped=skipped_lines,
                                   optimize_mode="max")
# Optimization starts ...
optimize_himme.run_optimization()

# Load the solutions:
with open("best_parameters.txt") as content: 

    for line in content:

        solution = np.array(line.split()).astype(np.float64)

    x, y = solution
    print("The Maximum found is  x = %.4f, y = %.4f" % (x, y))
