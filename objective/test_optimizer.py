import numpy as np
import matplotlib.pyplot as plt

class Rosenbrock():

    def __init__(self, x_ranges, y_ranges):

        self.x_limit = np.arange(x_ranges[0],x_ranges[1],x_ranges[-1])

        self.y_limit = np.arange(y_ranges[0],y_ranges[1],y_ranges[-1])

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

    def compute_z(self,parameters):

        sum_f_x = 0

        for i in range(0, parameters.size - 1):

            sum_f_x += ((1 - parameters[i] )**2 +
                        100*(parameters[i+1]- parameters[i]**2)**2)

        return (sum_f_x)**2

    def update(self, func_expand, best_func, status=None):

        pass

        return None 

    def optimize(self, para_type_lst, parameters, status=None):

        return self.compute_z(parameters)

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

        return (predict_val)**2 

    def optimize(self, para_type_lst, parameters, status=None):

        return self.compute_z(parameters) 

    def update(self, func_expand, best_func, status=None):

        pass

        return None 
