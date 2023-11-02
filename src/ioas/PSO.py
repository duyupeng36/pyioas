__all__ = ['PSO']


from datetime import datetime

import numpy as np
from ..base.baseoptimizer import BaseOptimizer


class PSO(BaseOptimizer):
    """
    Particle Swarm Optimization (PSO) algorithm.
    """

    def __init__(self, problem, population_size, maximum_iterations,
                 velocity_max=6,
                 weight_max=0.9, weight_min=0.2,
                 individual_coefficient=2, social_coefficient=2, **kwargs):
        super().__init__(problem, population_size, maximum_iterations, **kwargs)
        self.velocity_max = velocity_max
        self.weight_max = weight_max
        self.weight_min = weight_min
        self.individual_coefficient = individual_coefficient
        self.social_coefficient = social_coefficient

        # 初始化粒子速度
        self.individuals_velocity = np.random.uniform(-self.velocity_max, self.velocity_max,
                                                      (self.population_size, self.problem.dim))

        # 个体最优
        self.best_individual_position = np.zeros((self.population_size, self.problem.dim))

    def solve(self):
        self.solution.start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for i in range(self.population_size):
            fitness = self.problem(self.individual_positions[i], **self.kwargs)
            if fitness < self.individual_fitness[i]:
                self.individual_fitness[i] = fitness
                self.best_individual_position[i] = self.individual_positions[i]
            if self.individual_fitness[i] <= self.solution.best_fitness:
                self.solution.best_fitness = self.individual_fitness[i]
                self.solution.best_position = self.individual_positions[i]

        for itr in range(self.maximum_iterations):
            weight = self.weight_max - (self.weight_max - self.weight_min) * itr / self.maximum_iterations
            for i in range(self.population_size):
                r1 = np.random.rand(self.problem.dim)
                r2 = np.random.rand(self.problem.dim)
                # update velocity
                self.individuals_velocity[i, :] = (
                    weight * self.individuals_velocity[i, :]
                    + self.individual_coefficient
                    * r1 * (self.best_individual_position[i, :] - self.individual_positions[i, :])
                    + self.social_coefficient
                    * r2 * (self.solution.best_position - self.individual_positions[i, :])
                )
                self.individuals_velocity[i, :] = np.where(
                    self.individuals_velocity[i, :] > self.velocity_max,
                    self.velocity_max, self.individuals_velocity[i, :])
                self.individuals_velocity[i, :] = np.where(
                    self.individuals_velocity[i, :] < -self.velocity_max,
                    -self.velocity_max, self.individuals_velocity[i, :])
                # update position
                self.individual_positions[i, :] += self.individuals_velocity[i, :]
                self.individual_positions[i, :] = self.space_bound(self.individual_positions[i, :])
            for i in range(self.population_size):
                fitness = self.problem(self.individual_positions[i, :], **self.kwargs)
                if fitness < self.individual_fitness[i]:
                    self.individual_fitness[i] = fitness
                    self.best_individual_position[i, :] = self.individual_positions[i, :]
                if self.individual_fitness[i] < self.solution.best_fitness:
                    self.solution.best_fitness = self.individual_fitness[i]
                    self.solution.best_position = self.individual_positions[i, :]
            self.solution.iteration_curve[itr] = self.solution.best_fitness
        self.solution.end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return self.solution

