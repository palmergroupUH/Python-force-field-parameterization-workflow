# Python standard library:
import numpy as np
import logging
import sys
import os
import random

# Local library:
import optimizer.optimizer_mod

# Third-party libraries:

# a gradient-free optimizer: Nelder-Mead simplex algorithm:


class NelderMeadSimplex(optimizer.optimizer_mod.set_optimizer):

    def __init__(self,
                 input_files,
                 f_objective,
                 logname=None,
                 skipped=None,
                 output=None,
                 optimize_mode=None,
                 nm_type=None):

        # built-in restart/output filename:

        self.log_file = "log.restart"
        self.current_file = "current.restart"
        self.best_obj_file = "best_objective.txt"
        self.best_parameters_file = "best_parameters.txt"

        # Inherit the following from parent class:
        # "set_optimizer" in "optimizer_mod.py"

        super().__init__(input_files, logname=logname, skipped=skipped)

        # variables and methods Inherited from set_optimizer:

        # 0. dump frequecny self.dump_para:
        # 1. self.ptype_lst:
        # (list of string defining the type of parameters)

        # 1. guess parameters (self.guess_parameter)
        # 2. parameters to be fitted or fixed (self.fit_and_fix)
        # 3. index of guess parameters to be constrained
        # (self.bounds_fit_index)
        # 4. constraints bounds (self.bounds)
        # 5. the type of optimizer (self.optimizer_type)
        # 5. mode of Nelder-Mead simplex: "perturb" or "restart"
        # (self.optimizer_argument)
        # 6. contents of optimizer (self.optimizer_input)

        # Methods Inherited:
        # 1. self.constrain()
        # 2. self.group_fixed()
        # 3. self.dump_restart()
        # 4. self.dump_best_parameters()
        # 5. self.dump_best_objective()

        # computing objective function

        self.f_obj = f_objective

        if (optimize_mode is None):

            self.optimize_mode = "minimize"

        else:

            self.optimize_mode = optimize_mode

        # Define the following Nelder-Mead variables:
        # self.vertices_sorted: all vertices that have been sorted
        # self.func_vertices_sorted: all function values sorted at vertices
        # self.num_vertices: number of vertices
        # self.worst
        # self.best
        # self.lousy

        # set the destination folders addresses for output and restart files
        self.set_the_dumping_address(output)

        # check optimizer type and its mode:

        self.parse_Nelder_Mead_Input()

        # initialize simplex

        self.initialize_simplex()

        # default: adaptive nelder-mead simplex coefficient

        self.TransformationCoeff("adaptive")

        # print Nelder-Mead optimization initialization

        self.print_Nelder_Mead_simplex_log()

    # check the general input:

    def set_the_dumping_address(self, output):

        if (output is None):

            self.output_address = os.getcwd()

            self.restart_address = os.getcwd()

        else:

            self.output_address = os.path.join(output, "Output")

            self.restart_address = os.path.join(output, "Restart")

        return None

    def print_Nelder_Mead_simplex_log(self):

        # inherited from optimzier_mod:

        self.print_optimizer_log()

        # Add only Nelder Mead simplex related info

        if (self.optimizer_argument[1] == "perturb"):

            self.logger.info("%d step size is : \n"
                             % (self.stepsize.size))

            self.logger.info(" ".join(str(step)
                             for step in self.stepsize))

            self.logger.info(100*"-" + "\n\n")

            self.print_Nelder_Mead_simplex()

            self.logger.info(100*"-" + "\n\n")

        if (self.optimizer_argument[1] == "restart"):

            self.print_Nelder_Mead_simplex()

        return None

    def print_Nelder_Mead_simplex(self):

        self.logger.info("Objective function values"
                         "(sorted in ascending order"
                         " from left to right) : \n")

        self.logger.info(" ".join(str(para) for para in
                         self.func_vertices_sorted) + "\n")

        self.logger.info(100*"-" + "n\n")
        self.logger.info("All Nelder-Mead vertices "
                         "(sorted in ascending order"
                         "from top to down): \n")

        for i in range(self.num_vertices):

            self.logger.info("Vertex %d: " % (i+1) +
                             " ".join(str(para)
                                      for para in
                                      self.vertices_sorted[i, :]) + "\n")

        return None

    def Nelder_Mead_restart_output(self, itera):

        content_dict = self.optimizer_restart_content(itera,
                                                      self.best_vertex)

        keys_ary = np.array(list(content_dict.keys()))

        start = keys_ary.max() + 1

        content_dict[start] = " ".join(str(para) for para in
                                       self.func_vertices_sorted) + "\n\n"

        for i in range(self.num_vertices):

            content_dict[start + i+1] = " ".join(str(para) for para in
                                                 self.vertices_sorted[i, :]) +\
                                                 " \n\n"

        return content_dict

    def parse_Nelder_Mead_Input(self):

        if (self.optimizer_type == "Nelder-Mead"):

            # at least 1 argument needed for Nelder-Mead:
            # either restart or perturb

            if (len(self.optimizer_argument) == 1):

                self.logger.error("ERROR: Missing a mode argument:"
                                  "'Peturb' or 'restart'")

                sys.exit("Check errors in log file !")

            self.optimizer_mode = self.optimizer_argument[1]

        else:

            self.logger.error("Optimizer: %s not recognized ! "
                              "Please choose optimizer: "
                              "'Nelder-Mead' " % self.optimizer_type)

            sys.exit("Check errors in log file !")

    def check_Nelder_Mead_mode(self):

        # Check each Nelder-Mead mode:

        if (self.optimizer_argument[1] != "perturb" and
                self.optimizer_argument[1] != "restart"):

            self.logger.error("ERROR: only two modes allowed in "
                              "Nelder-Mead Simplex: "
                              "perturb or restart")

            sys.exit("Check errors in log file !")

        # Extract perturb argument:

        # if perturb used and at least two arguments are provided:

        if (self.optimizer_argument[1] == "perturb" and
                len(self.optimizer_argument) == 4):

            optimizer_mode_arg = self.optimizer_argument[2:]

            self.check_Nelder_Mead_perturb_mode(optimizer_mode_arg)

        elif (self.optimizer_argument[1] == "perturb" and
                len(self.optimizer_argument) == 2):

            self.check_provided_perturb_stepsize()

        # Extract restart argument

        if (self.optimizer_argument[1] == "restart"):

            self.check_restart_argument()

        return None

    def check_Nelder_Mead_perturb_mode(self, optimizer_mode_arg):

        # check 1st argument:

        if (optimizer_mode_arg[0] != "random" and
                optimizer_mode_arg[0] != "+" and
                optimizer_mode_arg[0] != "-"):

            self.logger.error("ERROR: If the 'perturb' mode is used,"
                              "its argument can only be: 'random', '+', '-'\n"
                              "The mode arugment: %s found in the "
                              "input file" % optimizer_mode_arg[0])

            sys.exit("Check errors in log file !")

        else:

            self.NM_perturb_mode = optimizer_mode_arg[0]

        # check 2nd argument:

        try:

            self.NM_perturb_stepsize = float(optimizer_mode_arg[1])

        except (ValueError, TypeError):

            self.logger.error("ERROR: When Nelder-Mead 'perturb' "
                              "mode is used, and arguments are "
                              "provided; The second argument "
                              "should be float ( percentage)")

            sys.exit("Check errors in log file !")

        return None

    def check_provided_perturb_stepsize(self):

        if (len(self.optimizer_input) != 1):

            self.logger.error("ERROR: If Nelder-Mead 'perturb' mode "
                              "is used,and no perturb arguments "
                              "provided,\n then,1 row of stepsize "
                              "( 0.1, -0.2, 0.8 -0.3 ... ) should "
                              "be provided by user\n"
                              "%d rows found in the input file"
                              % len(self.optimizer_input))

            sys.exit("Check errors in log file !")

        try:

            self.stepsize = (np.array(self.optimizer_input[0])
                               .astype(np.float64))

        except (ValueError, TypeError):

            self.logger.error("ERROR: Invalide perturbed stepsize encountered"
                              " when using Nelder-Mead 'perturb' mode\n")

            sys.exit("Check errors in log file !")

        return None

    def generate_simplex_stepsize(self):

        if (hasattr(self, "NM_perturb_mode") and
                hasattr(self, "NM_perturb_stepsize")):

            num_fitted = np.sum(self.fit_and_fix == 1)

            self.stepsize = np.zeros(num_fitted)

            for i in range(num_fitted):

                self.stepsize[i] = self.NM_perturb_stepsize *\
                                   self.perturb_sign(self.NM_perturb_mode)

        else:

            self.stepsize = (np.array(self.optimizer_input[0])
                               .astype(np.float64))

            # the number of perturbed vertices must be equal to the number
            # of fitted parameters

            if (self.stepsize.size != np.sum(self.fit_and_fix == 1)):

                self.logger.error("ERROR: When Nelder-Mead 'perturb'"
                                  " mode is used; The nubmer of "
                                  "perturbed stepsize must be equal "
                                  " to the nubmer of fitted "
                                  "parameters ( = 1 )")

                sys.exit("Check errors in log file !")

        return None

    def perturb_sign(self, mode):

        if (mode == "random"):

            if (random.uniform(0, 1) < 0.5):

                return 1

            else:

                return -1

        elif (mode == "+"):

            return 1

        elif (mode == "-"):

            return -1

        else:

            self.logger.error("perturb sign mode not recgonized "
                              "! Please choose '+','-', or 'random'")

            sys.exit("Check errors in log file !")

    def check_restart_argument(self):

        if (len(self.optimizer_input) < 3):

            self.logger.error("ERROR: When Nelder-Mead restart mode "
                              "is used \n At least "
                              "3 rows of arguments:"
                              "\n ( 1st row: objective functions, "
                              "2nd row: first vertex, "
                              "3rd row: second vertex ... \n"
                              "%d rows found in the input file"
                              % len(self.optimizer_input))

            sys.exit("Check errors in log file !")

        # check size of objective functions and vertices provided:

        number_vertices = len(self.optimizer_input[0])

        if (len(self.optimizer_input[1:]) != number_vertices):

            self.logger.error("ERROR: When Nelder-Mead restart mode "
                              "is used: \n "
                              "Number of vertices should be equal "
                              "to number of vertex parameter")

            sys.exit("Check errors in log file !")

        # check consistency of parameters with number of vertices

        for i in range(len(self.optimizer_input[1:])):

            if (len(self.optimizer_input[i+1]) != number_vertices - 1):

                self.logger.error("ERROR: When Nelder-Mead restart "
                                  "mode is used: \n"
                                  "Number parameters should be "
                                  "( number_vertices - 1 )")

                sys.exit("Check errors in log file !")

        return None

    def parse_existing_simplex(self):

        try:

            f_vertices = np.array(self.optimizer_input[0]).astype(np.float64)

            self.num_vertices = f_vertices.size

            self.num_fitting = self.num_vertices - 1

            vertices_mat = np.zeros((self.num_vertices,
                                    self.num_fitting),
                                    dtype=np.float64)

            for i in range(len(self.optimizer_input[1:])):

                vertices_mat[i, :] = (np.array(self.optimizer_input[i+1])
                                        .astype(np.float64))

            self.sort_simplex(f_vertices, vertices_mat)

        except (ValueError, TypeError):

            self.logger.error("ERROR: When Nelder-Mead restart mode "
                              "is used: ValueError or TypeError"
                              "encountered in reading the "
                              "restart simplex")

            sys.exit("Check errors in log file !")

        return None

    def initialize_simplex(self):

        # choose the two modes:

        self.check_Nelder_Mead_mode()

        # "perturb" create the new simplex

        if (self.optimizer_mode == "perturb"):

            # either read or create step size based on the input

            self.generate_simplex_stepsize()

            vertices_mat = self.generate_simplex("orthogonal")

            func_vertices = self.compute_func_vertices(vertices_mat,
                                                       choice="guess")

            self.sort_simplex(func_vertices, vertices_mat)

        # "restart" uses the existing simplex

        elif (self.optimizer_mode == "restart"):

            self.parse_existing_simplex()

        return None

    def generate_simplex(self, simplex_type):

        # "1" is the fitted variable

        self.gussed_fitting_para = self.guess_parameter[self.fit_and_fix == 1]

        self.num_fitting = self.gussed_fitting_para.size

        # "0" is the fixed variable

        self.fix_variable = self.guess_parameter[self.fit_and_fix == 0]

        self.num_fixed = self.fix_variable.size

        # Number of vertices: n + 1 (n is the number of fitting parameters)

        self.num_vertices = self.num_fitting + 1

        # generate orthogonal simplex

        if (simplex_type == "orthogonal"):

            vertices_mat = self.use_orthogonal_simplex()

        return vertices_mat

    def use_orthogonal_simplex(self):

        # initialize an array: "vertices_mat" to save all vertices
        # matrix Format:
        # vertex 1: [ para1, para2 ... para n ]
        # vertex 2: [ para1, para2 ... para n  ]
        # ...
        # vertex n+1: [ para1, para2 ... para n  ]
        # So vertices_mat has dimension of ( n+1, n )

        vertices_mat = np.zeros((self.num_vertices, self.num_fitting))

        # first vertex is the guess parameter

        vertices_mat[0, :] = self.gussed_fitting_para

        # orthogonal perturbation of vertices

        shift_vector = np.eye(self.num_fitting)

        # loop over vertex except first one which is the guess parameter

        for i in range(1, self.num_fitting+1):

            # if the parameter its self is 0, then add 0.005 to it

            if (self.gussed_fitting_para[i-1] == 0):

                new_vertices = self.gussed_fitting_para + \
                               self.stepsize[i-1] * \
                               shift_vector[i-1, :]*0.005

            else:

                new_vertices = self.gussed_fitting_para + \
                               self.stepsize[i-1]*shift_vector[i-1, :] * \
                               self.gussed_fitting_para[i-1]

            self.constrain(new_vertices)

            vertices_mat[i, :] = new_vertices

        return vertices_mat

    def sort_simplex(self, func_vertices, vertices_mat):

        # For minimization problems:
        # sort the objective function from small to large
        # So, best_objective function is the minima of all objective function

        if (self.optimize_mode == "minimize"):

            self.best_indx = 0

            self.worst_indx = -1

            self.lousy_indx = -2

        elif (self.optimize_mode == "maximize"):

            self.best_indx = -1

            self.lousy_indx = 1

            self.worst_indx = 0

        else:

            self.logger.error("ERROR: optimize mode can only "
                              "be either 'minimize' or 'maximize'")

            sys.exit("Check errors in log file !")

        # argsort default sort order is the asscending order

        ascending = np.argsort(func_vertices)

        self.vertices_sorted = vertices_mat[ascending, :]

        self.func_vertices_sorted = func_vertices[ascending]

        # best,worst and lousy objective function values

        self.best = self.func_vertices_sorted[self.best_indx]

        self.worst = self.func_vertices_sorted[self.worst_indx]

        self.lousy = self.func_vertices_sorted[self.lousy_indx]

        # best,worst and lousy vertex

        self.best_vertex = self.vertices_sorted[self.best_indx, :]

        self.worst_vertex = self.vertices_sorted[self.worst_indx, :]

        self.lousy_vertex = self.vertices_sorted[self.lousy_indx, :]

        return None

    def TransformationCoeff(self, keyword):

        if (keyword == "standard"):

            self.alpha = 1.0

            self.kai = 2.0

            self.gamma = 0.5

            self.sigma = 0.5

        elif (keyword == "adaptive"):

            self.alpha = 1.0

            self.kai = 1 + 2.0/self.num_fitting

            self.gamma = 0.75 - 1.0/(2*self.num_fitting)

            self.sigma = 1.0 - 1.0/self.num_fitting

    def compute_func_vertices(self, vertices_mat, choice):

        num_vertices = vertices_mat[:, 0].size

        f_vertices = np.zeros(num_vertices, dtype=np.float64)

        # get the temporary best
        if (choice == "shrink"):

            self.temp_best = self.best

        for i in range(num_vertices):

            in_parameters = vertices_mat[i, :]

            self.constrain(in_parameters)

            in_parameters_full = self.group_fixed(in_parameters)

            if (choice == "guess"):

                # guess parameters:
                if (i == 0):

                    f_vertices[i] = self.f_obj.optimize(self.ptype_lst[0],
                                                        in_parameters_full,
                                                        status="guess")
                else:

                    f_vertices[i] = self.f_obj.optimize(self.ptype_lst[0],
                                                        in_parameters_full,
                                                        status="new")

            elif (choice == "shrink"):

                current_obj = self.f_obj.optimize(self.ptype_lst[0],
                                                  in_parameters_full,
                                                  status="new")

                self.f_obj.update(current_obj, self.temp_best, status="new")

                if (current_obj < self.temp_best):

                    # update the current best
                    self.temp_best = current_obj

                f_vertices[i] = current_obj

        return f_vertices

    def check_convergence_status(self, n_iteration):

        self.logger.info("Current iteration: %d "
                         "finishes \n\n" % n_iteration)

        self.logger.info(30*"-" + "Optimization status: " +
                         30*"-" + "\n")

        self.logger.info("Current Best objective: "
                         "%.10f\n\n" % self.best)

        self.logger.info("Current Best parameters: " +
                         " ".join(str(para) for para in
                                  self.vertices_sorted[self.best_indx, :]) +
                         "\n\n")

        self.logger.info("Current Worst objective: "
                         "%.10f\n\n" % self.worst)

        self.logger.info("Current Worst parameters: " +
                         " ".join(str(para) for para in
                                  self.vertices_sorted[self.worst_indx, :]) +
                         "\n")

        self.logger.info(70*"-" + "\n")

        converged = self.termination_criterion_is_met(n_iteration)

        if (not converged):

            self.logger.info("Then, start next iteration ... \n")

        return converged

    def Centroid(self):

        # select all vertices except the worst vertex

        except_worst = self.vertices_sorted[:self.worst_indx, :]

        self.logger.info("Compute the centroid  ...\n")

        # compute the geometric center

        return np.mean(except_worst, axis=0)

    def Reflect(self, centroid):

        reflected_vetertex = centroid + self.alpha*(centroid -
                                                    self.worst_vertex)

        self.constrain(reflected_vetertex)

        self.logger.info("Perform reflection ... \n")

        return reflected_vetertex

    def Accept(self, vertex, func_vertex, transform_keyword):

        # subsitude worst vertex

        self.vertices_sorted[self.worst_indx, :] = vertex

        self.func_vertices_sorted[self.worst_indx] = func_vertex

        self.logger.info("%s is accepted ... \n" % transform_keyword)

    def Expand(self, reflected, centroid):

        expanded_vertex = centroid + self.kai*(reflected - centroid)

        self.constrain(expanded_vertex)

        self.logger.info("Perform expansion to further explore "
                         "the reflected direction ... \n")

        return expanded_vertex

    def Outside_Contract(self, centroid, reflected_vertex):

        outside_vertex = centroid + self.gamma*(reflected_vertex - centroid)

        self.constrain(outside_vertex)

        self.logger.info("Reflected vertex is in between "
                         "second-worst vertex and worst vertex ...\n"
                         "Perform outside contraction ... \n\n")

        return outside_vertex

    def Inside_Contract(self, centroid):

        inside_vertex = centroid + self.gamma*(self.worst_vertex - centroid)

        self.constrain(inside_vertex)

        self.logger.info("Reflected vertex is worst than that "
                         "of the worst vertex ...\n\n"
                         "Perform inside contraction ... \n\n")

        return inside_vertex

    def Shrink(self):

        shrinked_vertices = np.zeros((self.num_vertices-1,
                                     self.num_vertices-1),
                                     dtype=np.float64)

        for i in range(self.num_vertices - 1):

            shrinked_vertex = (self.best_vertex +
                               self.sigma *
                               (self.vertices_sorted[i+1, :] -
                                self.best_vertex))

            self.constrain(shrinked_vertex)

            shrinked_vertices[i, :] = shrinked_vertex

        self.logger.info("The contracted vertex ( outisde/insdie )"
                         "is worse than the worst vertex ...\n\n"
                         "Perform shrinkage ... \n\n")

        func_vertices = self.compute_func_vertices(shrinked_vertices,
                                                   choice="shrink")

        self.vertices_sorted[self.best_indx+1:, :] = shrinked_vertices

        self.func_vertices_sorted[self.best_indx+1:] = func_vertices

        self.sort_simplex(self.func_vertices_sorted, self.vertices_sorted)

        return None

# =============================================================================
#                            Nelder Mead Simplex Algorithm
# =============================================================================

    def run_optimization(self):

        # set converged status False to start iteration:

        self.converged = False

        # Nelder Mead simplex algorithm:

        for itera in range(self.max_iteration):

            # terminate the optimization if
            # "self.check_convergence_status" returns True:

            if (self.converged):

                break

            self.logger.info(17*"===="+"\n")

            self.logger.info("Current iteration: %d starts \n\n" % itera)

            # Centroid

            centroid = self.Centroid()

            # Reflection

            r_vertex = self.Reflect(centroid)

            f_r_vertex = self.f_obj.optimize(self.ptype_lst[0],
                                             self.group_fixed(r_vertex),
                                             status="old")

            if (self.best <= f_r_vertex < self.lousy):

                self.Accept(r_vertex, f_r_vertex, "Reflection")

                self.sort_simplex(self.func_vertices_sorted,
                                  self.vertices_sorted)

                self.converged = self.check_convergence_status(itera)

                self.optimization_output(itera)

                continue

            # Expansion

            if (f_r_vertex < self.best):

                e_vertex = self.Expand(r_vertex, centroid)

                func_expand = self.f_obj.optimize(self.ptype_lst[0],
                                                  self.group_fixed(e_vertex),
                                                  status="new")

                if (func_expand < f_r_vertex):

                    self.f_obj.update(func_expand, self.best, status="new")

                    self.Accept(e_vertex,
                                func_expand,
                                "Expansion")

                    self.sort_simplex(self.func_vertices_sorted,
                                      self.vertices_sorted)

                    self.converged = self.check_convergence_status(itera)

                    self.optimization_output(itera)

                    continue

                else:

                    self.f_obj.update(f_r_vertex,
                                      self.best,
                                      status="old")

                    self.Accept(r_vertex,
                                f_r_vertex,
                                "Reflection")

                    self.sort_simplex(self.func_vertices_sorted,
                                      self.vertices_sorted)

                    self.converged = self.check_convergence_status(itera)

                    self.optimization_output(itera)

                    continue

            # Contraction

            # outside contraction:

            if (f_r_vertex >= self.lousy):

                if (self.lousy <= f_r_vertex < self.worst):

                    o_vertex = self.Outside_Contract(centroid, r_vertex)

                    f_o_vertex = self.f_obj.optimize(self.ptype_lst[0],
                                                     self.group_fixed(o_vertex),
                                                     status="new")

                    if (f_o_vertex <= f_r_vertex):

                        self.f_obj.update(f_o_vertex,
                                          self.best,
                                          status="new")

                        self.Accept(o_vertex,
                                    f_o_vertex,
                                    "Outside contraction")

                        self.sort_simplex(self.func_vertices_sorted,
                                          self.vertices_sorted)

                        self.converged = self.check_convergence_status(itera)

                        self.optimization_output(itera)

                        continue

                    else:

                        self.Shrink()

                        self.converged = self.check_convergence_status(itera)

                        self.optimization_output(itera)

                        continue

                # inside contraction:

                if (f_r_vertex >= self.worst):

                    i_vertex = self.Inside_Contract(centroid)

                    f_i_vertex = self.f_obj.optimize(self.ptype_lst[0],
                                                     self.group_fixed(i_vertex),
                                                     status="new")

                    if (f_i_vertex < self.worst):

                        self.f_obj.update(f_i_vertex,
                                          self.best,
                                          status="new")

                        self.Accept(i_vertex,
                                    f_i_vertex,
                                    "Inside contraction")

                        self.sort_simplex(self.func_vertices_sorted,
                                          self.vertices_sorted)

                        self.converged = self.check_convergence_status(itera)

                        self.optimization_output(itera)

                        continue

                    else:

                        self.Shrink()

                        self.converged = self.check_convergence_status(itera)

                        self.optimization_output(itera)

                        continue

# =============================================================================
#                             Termination criterion
# =============================================================================

    def termination_criterion_is_met(self, n_itera):

        self.logger.debug("Class NelderMeadSimplex:terminate "
                          "function entered successfully !")

        if (np.amin(self.func_vertices_sorted) == 0):

            self.logger.info("Convergence criterion 1 is met: "
                             "minimum of objective function "
                             "is equal to 0")

            self.logger.info("Optimization converges "
                             "and program exits ! \n")

            return True

        if ((np.amax(self.func_vertices_sorted) /
                np.amin(self.func_vertices_sorted)-1) <
                self.obj_tol):

            sci_obj = "{0:.1e}".format(self.obj_tol)

            self.logger.info("Convergence criterion 2 is met: "
                             "Ratio of obj_max/obj_min -1  "
                             "< %s !\n" % sci_obj)

            self.logger.info("Optimization converges "
                             "and program exits ! \n")

            return True

        unique_obj, repeat = np.unique(self.func_vertices_sorted,
                                       return_counts=True)

        if (unique_obj.size < self.func_vertices_sorted.size):

            self.logger.info("Convergence criterion 3 is met: "
                             "some objective functions of "
                             "different vertex begin to converge")

            self.logger.info(" ".join(str(obj) for obj
                                      in self.vertices_sorted))

            self.logger.info("The objective function values "
                             "for all vertex are: \n ")

            self.logger.info(" ".join(str(obj) for obj
                                      in self.func_vertices_sorted))

            self.logger.info("Optimization converges and "
                             "program exits ! \n")

            return True

        if (np.all(np.std(self.func_vertices_sorted) < self.para_tol)):

            sci_para = "{0:.1e}".format(self.para_tol)

            self.logger.info("Convergence criterion 4 is met: the "
                             "standard deviation of force-field "
                             "paramteters across all vertices "
                             "is %s  !\n" % sci_para)

            self.logger.info("Optimization converges and "
                             "program exits ... \n")

            return True

        if (n_itera+1 == self.max_iteration):

            self.logger.info("Convergence criterion 5 is met: "
                             "Maximum number of iteration "
                             "is reached !\n")

            self.logger.info("Maximum iteration %d is reached and "
                             "Program exit !" % self.max_iteration)

            return True

        self.logger.debug("Class NelderMeadSimplex:"
                          "terminate function exit successfully !")

        return None

# =============================================================================
#                                      Output
# =============================================================================

    # dictionary based output format:
    #   --keys: line number
    #   --values: content ( string )

    def optimization_output(self, itera):

        # Add output functions here

        content_dict = self.Nelder_Mead_restart_output(itera)

        # inherited from optimizer_mod
        self.dump_restart(itera, [self.log_file,
                                  self.current_file],
                          self.restart_address,
                          content_dict)

        # inherited from optimizer_mod
        self.dump_best_objective(itera,
                                 self.best_obj_file,
                                 self.output_address,
                                 self.best)

        # inherited from optimizer_mod
        self.dump_best_parameters(itera,
                                  self.best_parameters_file,
                                  self.output_address,
                                  self.best_vertex)

        return None

    def dump_current_simplex(self, itera):

        simplex = {}

        for iv in range(self.num_vertices):

            simplex[iv] = " ".join(str(para) for para in
                                   self.vertices_sorted[iv, :]) + "\n"

        simplex[self.num_vertices] = " ".join(str(para) for para in
                                              self.vertices_sorted[0, :]) + "\n"

        simplex_file = "simplex_%d.txt" % itera

        self.write_optimizer_output(self.output_freq,
                                    itera,
                                    self.output_address,
                                    simplex_file,
                                    "w",
                                    simplex)

        return None
