from datetime import datetime

import numpy as np
from numpy.random import permutation as randperm

from ..base.baseoptimizer import BaseOptimizer


class SOA(BaseOptimizer):
    """
    Seagull optimization algorithm: Theory and its applications for large-scale industrial engineering problems
    """
    def __init__(self, problem, population_size, maximum_iterations, fc=2, u=1, v=1, **kwargs):
        super().__init__(problem, population_size, maximum_iterations, **kwargs)
        self.fc = fc
        self.u = u
        self.v = v

    def solve(self):
        self.solution.start_time = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
        for i in range(self.population_size):
            self.individual_fitness[i] = self.problem(self.individual_positions[i])
        for i in range(self.population_size):
            if self.individual_fitness[i] < self.solution.best_fitness:
                self.solution.best_fitness = self.individual_fitness[i]
                self.solution.best_position = self.individual_positions[i]

        for itr in range(self.maximum_iterations):

            A = self.fc - itr * (self.fc / self.maximum_iterations)
            B = 2 * A * np.random.rand()
            for i in range(self.population_size):
                C_s = A * self.individual_positions[i]
                M_s = B * (self.solution.best_position - self.individual_positions[i])
                D_s = abs(C_s + M_s)

                k_x = np.random.uniform(0, 2 * np.pi)
                r_x = self.u * np.exp(k_x * self.v)
                x = r_x * np.sin(k_x)

                k_y = np.random.uniform(0, 2 * np.pi)
                r_y = self.u * np.exp(k_y * self.v)
                y = r_y * np.sin(k_y)

                k_z = np.random.uniform(0, 2 * np.pi)
                r_z = self.u * np.exp(k_z * self.v)
                z = r_z * k_z

                new_individual_position = D_s * x * y * z + self.solution.best_position
                new_individual_position = self.space_bound(new_individual_position)
                fitness = self.problem(new_individual_position)
                if fitness < self.individual_fitness[i]:
                    self.individual_fitness[i] = fitness
                    self.individual_positions[i] = new_individual_position
            for i in range(self.population_size):
                if self.individual_fitness[i] < self.solution.best_fitness:
                    self.solution.best_fitness = self.individual_fitness[i]
                    self.solution.best_position = self.individual_positions[i]

            self.solution.iteration_curve[itr] = self.solution.best_fitness
        self.solution.end_time = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
        return self.solution
