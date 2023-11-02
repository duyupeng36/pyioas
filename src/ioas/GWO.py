from datetime import datetime

import numpy as np
from numpy.random import permutation as randperm

from ..base.baseoptimizer import BaseOptimizer


class GWO(BaseOptimizer):
    """
    Grey Wolf Optimizer
    """

    def __init__(self, problem, population_size, maximum_iterations, **kwargs):
        super().__init__(problem, population_size, maximum_iterations, **kwargs)

        self.alpha_position = np.zeros(self.problem.dim)
        self.alpha_fitness = np.inf
        self.beta_position = np.zeros(self.problem.dim)
        self.beta_fitness = np.inf
        self.delta_position = np.zeros(self.problem.dim)
        self.delta_fitness = np.inf

    def solve(self):
        self.solution.start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for itr in range(self.maximum_iterations):
            for i in range(self.population_size):
                fitness = self.problem(self.individual_positions[i])
                if fitness < self.alpha_fitness:
                    self.delta_fitness = self.beta_fitness
                    self.delta_position = self.beta_position.copy()
                    self.beta_fitness = self.alpha_fitness
                    self.beta_position = self.alpha_position.copy()
                    self.alpha_fitness = fitness
                    self.alpha_position = self.individual_positions[i].copy()
                    self.solution.best_fitness = self.alpha_fitness
                    self.solution.best_position = self.alpha_position.copy()
                if self.alpha_fitness < fitness < self.beta_fitness:
                    self.delta_fitness = self.beta_fitness
                    self.delta_position = self.beta_position.copy()
                    self.beta_fitness = fitness
                    self.beta_position = self.individual_positions[i].copy()

                if self.alpha_fitness < fitness < self.delta_fitness and fitness > self.beta_fitness:
                    self.delta_fitness = fitness
                    self.delta_position = self.individual_positions[i].copy()

            a = 2 - itr * (2 / self.maximum_iterations)
            for i in range(self.population_size):
                r1 = np.random.rand(self.problem.dim)
                r2 = np.random.rand(self.problem.dim)

                A1 = 2 * a * r1 - a
                C1 = 2 * r2

                D_alpha = np.abs(C1 * self.alpha_position - self.individual_positions[i])
                X1 = self.alpha_position - A1 * D_alpha

                r1 = np.random.rand(self.problem.dim)
                r2 = np.random.rand(self.problem.dim)

                A2 = 2 * a * r1 - a
                C2 = 2 * r2

                D_beta = np.abs(C2 * self.beta_position - self.individual_positions[i])
                X2 = self.beta_position - A2 * D_beta

                r1 = np.random.rand(self.problem.dim)
                r2 = np.random.rand(self.problem.dim)

                A3 = 2 * a * r1 - a
                C3 = 2 * r2

                D_delta = np.abs(C3 * self.delta_position - self.individual_positions[i])
                X3 = self.delta_position - A3 * D_delta

                self.individual_positions[i] = (X1 + X2 + X3) / 3

            self.solution.iteration_curve[itr] = self.alpha_fitness
        # 可视化
        self.solution.end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return self.solution

