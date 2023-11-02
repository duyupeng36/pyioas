__all__ = ['QIO']

from datetime import datetime

import numpy as np

from ..base.baseoptimizer import BaseOptimizer
from ..base.problem import BaseProblem
from ..base.utils import GQI


class QIO(BaseOptimizer):
    """
    Quadratic Interpolation Optimization (QIO): A new optimization algorithm based on generalized quadratic interpolation and its applications to real-world engineering problems
    """

    def __init__(self, problem: BaseProblem, population_size: int, maximum_iterations: int, **kwargs):
        super().__init__(problem, population_size, maximum_iterations, **kwargs)

    def solve(self):
        self.solution.start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for i in range(self.population_size):
            self.individual_fitness[i] = self.problem(self.individual_positions[i], **self.kwargs)
        for i in range(self.population_size):
            if self.individual_fitness[i] <= self.solution.best_fitness:
                self.solution.best_fitness = self.individual_fitness[i]
                self.solution.best_position = self.individual_positions[i].copy()

        for itr in range(self.maximum_iterations):
            new_individual_position = np.zeros(self.problem.dim)
            for i in range(self.population_size):
                K = [j for j in range(i)] + [j for j in range(i + 1, self.population_size)]
                if np.random.rand() < 0.5:
                    rand_index = np.random.choice(np.arange(0, self.population_size - 1), 3, replace=False)
                    K1 = K[rand_index[0]]
                    K2 = K[rand_index[1]]
                    K3 = K[rand_index[2]]
                    f1 = self.individual_fitness[K1]
                    f2 = self.individual_fitness[K2]
                    f3 = self.individual_fitness[i]
                    for j in range(self.problem.dim):
                        x1 = self.individual_positions[K1, j]
                        x2 = self.individual_positions[K2, j]
                        x3 = self.individual_positions[i, j]
                        new_individual_position[j] = GQI(x1, x2, x3, f1, f2, f3, self.problem.lower_boundary[j], self.problem.upper_boundary[j])
                    a = np.cos(np.pi / 2 * itr / self.maximum_iterations)
                    b = 0.7 * a + 0.15 * a * (np.cos(5 * np.pi * itr / self.maximum_iterations) + 1)
                    w1 = 3 * b * np.random.randn()
                    new_individual_position = new_individual_position + w1 * (self.individual_positions[K3] - new_individual_position) + np.round(0.5 * (0.05 + np.random.randn())) * np.log(np.random.rand() / np.random.rand())
                else:
                    rand_index = np.random.choice(np.arange(0, self.population_size - 1), 2, replace=False)
                    K1 = K[rand_index[0]]
                    K2 = K[rand_index[1]]
                    f1 = self.individual_fitness[K1]
                    f2 = self.individual_fitness[K2]
                    f3 = self.individual_fitness[i]
                    for j in range(self.problem.dim):
                        x1 = self.individual_positions[K1, j]
                        x2 = self.individual_positions[K2, j]
                        x3 = self.individual_positions[i, j]
                        new_individual_position[j] = GQI(x1, x2, x3, f1, f2, f3, self.problem.lower_boundary[j],
                                                         self.problem.upper_boundary[j])
                    w2 = 3 * (1 - (itr - 1) / self.maximum_iterations) * np.random.randn()
                    rD = np.random.randint(0, self.problem.dim)
                    new_individual_position = new_individual_position + w2 * (self.solution.best_position - np.round(1 + np.random.rand()) * (self.problem.upper_boundary - self.problem.lower_boundary) / (self.problem.upper_boundary[rD] - self.problem.lower_boundary[rD]) * self.individual_positions[i, rD])
                new_individual_position = self.space_bound(new_individual_position)
                fitness = self.problem(new_individual_position)
                if fitness < self.individual_fitness[i]:
                    self.individual_fitness[i] = fitness
                    self.individual_positions[i] = new_individual_position.copy()

            for i in range(self.population_size):
                if self.individual_fitness[i] < self.solution.best_fitness:
                    self.solution.best_fitness = self.individual_fitness[i]
                    self.solution.best_position = self.individual_positions[i].copy()
            self.solution.iteration_curve[itr] = self.solution.best_fitness

        self.solution.end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return self.solution
