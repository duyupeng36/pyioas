from datetime import datetime

import numpy as np
from numpy.random import permutation as randperm

from ..base import BaseOptimizer


class AHA(BaseOptimizer):

    def __init__(self, problem, population_size, maximum_iterations, **kwargs):
        super().__init__(problem, population_size, maximum_iterations, **kwargs)

        self.visited_table = np.zeros((population_size, population_size))
        self.diag_ind = np.diag_indices(population_size)
        self.visited_table[self.diag_ind] = float('inf')

    def solve(self):
        self.solution.start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        for i in range(self.population_size):
            self.individual_fitness[i] = self.problem(self.individual_positions[i, :])
            if self.individual_fitness[i] < self.solution.best_fitness:
                self.solution.best_fitness = self.individual_fitness[i]
                self.solution.best_position = self.individual_positions[i].copy()

        for itr in range(self.maximum_iterations):
            self.visited_table[self.diag_ind] = float('-inf')
            for i in range(self.population_size):
                vector = self.direct_vector(i)
                

    def direct_vector(self, ind):
        """
        flight pattern

        Parameters
        -----------
        ind: int
            Individual index
        """
        vector = np.zeros((self.population_size, self.problem.dim))
        r = np.random.rand()
        # Diagonal flight
        if r < 1 / 3:
            rand_dim = randperm(self.problem.dim)
            if self.problem.dim >= 3:
                rand_num = np.ceil(np.random.rand() * (self.problem.dim - 2))
            else:
                rand_num = np.ceil(np.random.rand() * (self.problem.dim - 1))

            vector[ind, rand_dim[:int(rand_num)]] = 1
        # Omnidirectional flight
        elif r > 2 / 3:
            vector[ind, :] = 1
        else:
            # Axial flight
            rand_num = np.ceil(np.random.rand() * (self.problem.dim - 1))
            vector[ind, int(rand_num)] = 1
        return vector

