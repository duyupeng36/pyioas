
from datetime import datetime

import numpy as np

from ..base.baseoptimizer import BaseOptimizer


class SMA(BaseOptimizer):
    """
    Slime mould algorithm: Anew method for stochastic optimization
    """
    EPSILON = 1E-10

    def __init__(self, problem, population_size, maximum_iterations, z=0.03, **kwargs):
        super().__init__(problem, population_size, maximum_iterations, **kwargs)
        self.weight = np.zeros((self.population_size, self.problem.dim))
        self.z = z

    def solve(self):
        self.solution.start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for i in range(self.population_size):
            self.individual_fitness[i] = self.problem(self.individual_positions[i, :], **self.kwargs)
            if self.individual_fitness[i] < self.solution.best_fitness:
                self.solution.best_fitness = self.individual_fitness[i]
                self.solution.best_position = self.individual_positions[i, :].copy()

        self.sort()
        for itr in range(self.maximum_iterations):
            worst_fitness = self.individual_fitness[-1]
            best_fitness = self.individual_fitness[1]
            self.weight[:self.population_size // 2, :] = 1 + np.log((best_fitness - self.individual_fitness[:self.population_size // 2]) / (best_fitness - worst_fitness + self.EPSILON) + 1).reshape((-1, 1)) @ np.random.rand(self.problem.dim).reshape((1, -1))
            self.weight[self.population_size // 2:, :] = 1 - np.log((self.individual_fitness[self.population_size // 2:] - worst_fitness) / (best_fitness - worst_fitness + self.EPSILON) + 1).reshape((-1, 1)) @ np.random.rand(self.problem.dim).reshape((1, -1))
            t = 1 - itr / self.maximum_iterations
            a = np.arctanh(t) if t != -1 and t != 1 else 1
            b = 1 - itr / self.maximum_iterations
            for i in range(self.population_size):
                if np.random.rand() < self.z:
                    self.individual_positions[i, :] = np.random.rand() * (self.problem.upper_boundary - self.problem.lower_boundary) + self.problem.lower_boundary
                else:
                    p = np.tanh(np.abs(self.individual_fitness[i] - self.solution.best_fitness))
                    vb = 2 * a * np.random.rand(self.problem.dim) - a
                    vc = 2 * b * np.random.rand(self.problem.dim) - b
                    if np.random.rand() < p:
                        A = np.random.randint(0, self.population_size)
                        B = np.random.randint(0, self.population_size)
                        self.individual_positions[i, :] = self.solution.best_position + vb * (self.weight[i, :] * self.individual_positions[A, :] - self.individual_positions[B, :])
                    else:
                        self.individual_positions[i, :] = vc * self.individual_positions[i, :]
                self.individual_positions[i] = self.space_bound(self.individual_positions[i])
                self.individual_fitness[i] = self.problem(self.individual_positions[i, :], **self.kwargs)
            self.sort()
            if self.individual_fitness[0] < self.solution.best_fitness:
                self.solution.best_fitness = self.individual_fitness[0]
                self.solution.best_position = self.individual_positions[0, :].copy()
            self.solution.iteration_curve[itr] = self.solution.best_fitness
        self.solution.end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return self.solution

    def sort(self):
        indices = self.individual_fitness.argsort()
        self.individual_fitness = self.individual_fitness[indices]
        self.individual_positions = self.individual_positions[indices, :]


