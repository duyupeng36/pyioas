from datetime import datetime

import numpy as np
from numpy.random import permutation as randperm

from ..base.baseoptimizer import BaseOptimizer


class AHA(BaseOptimizer):
    """
    Artificial hummingbird algorithm: A new bio-inspired optimizer with its engineering applications
    """

    def __init__(self, problem, population_size, maximum_iterations, **kwargs):
        super().__init__(problem, population_size, maximum_iterations, **kwargs)

        self.visited_table = np.zeros((population_size, population_size))
        self.diag_ind = np.diag_indices(population_size)
        self.visited_table[self.diag_ind] = float('inf')

    def solve(self):
        self.solution.start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        for i in range(self.population_size):
            self.individual_fitness[i] = self.problem(self.individual_positions[i, :])

        for itr in range(self.maximum_iterations):
            self.visited_table[self.diag_ind] = float('-inf')
            for i in range(self.population_size):
                vector = self.direct_vector(i)
                if np.random.rand() < 0.05:
                    max_unvisited_time = np.max(self.visited_table[i, :])
                    target_food_index = self.visited_table[i, :].argmax()
                    mut_ind = np.where(self.visited_table[i, :] == max_unvisited_time)
                    if len(mut_ind[0]) > 1:
                        ind = self.individual_fitness[mut_ind].argmin()
                        target_food_index = mut_ind[0][ind]
                    new_individual_position = self.individual_positions[target_food_index, :] + np.random.randn() * vector[i, :] * (self.individual_positions[i, :] - self.individual_positions[target_food_index, :])
                    new_individual_position = self.space_bound(new_individual_position)
                    fitness = self.problem(new_individual_position, **self.kwargs)
                    if fitness < self.individual_fitness[i]:
                        self.individual_positions[i, :] = new_individual_position
                        self.individual_fitness[i] = fitness
                        self.visited_table[i, target_food_index] = 0
                        self.visited_table[:, i] = np.max(self.visited_table, axis=1) + 1
                        self.visited_table[i, i] = float('-inf')
                    else:
                        self.visited_table[i, :] += 1
                        self.visited_table[i, target_food_index] = 0
                else:
                    new_individual_position = self.individual_positions[i, :] + np.random.randn() * vector[i, :] * self.individual_positions[i, :]
                    new_individual_position = self.space_bound(new_individual_position)
                    fitness = self.problem(new_individual_position, **self.kwargs)
                    if fitness < self.individual_fitness[i]:
                        self.individual_positions[i, :] = new_individual_position
                        self.individual_fitness[i] = fitness
                        self.visited_table[i, :] += 1
                        self.visited_table[:, i] = np.max(self.visited_table, axis=1) + 1
                        self.visited_table[i, i] = float('-inf')
                    else:
                        self.visited_table[i, :] += 1
            self.visited_table[self.diag_ind] = float('nan')
            if np.mod(itr, 2 * self.population_size) == 0:
                self.visited_table[self.diag_ind] = float('inf')
                migration_index = self.individual_fitness.argmax()
                self.individual_positions[migration_index, :] = np.random.rand() * (self.problem.upper_boundary - self.problem.lower_boundary) + self.problem.lower_boundary
                self.individual_fitness[migration_index] = self.problem(self.individual_positions[migration_index, :])
                self.visited_table[migration_index, :] += 1
                self.visited_table[:, migration_index] = np.max(self.visited_table, axis=1) + 1
                self.visited_table[migration_index, migration_index] = float('-inf')
            self.visited_table[self.diag_ind] = float('nan')

            self.update(itr)
            # visualization

        self.solution.end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return self.solution

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

