
__all__ = ['HHO']

from datetime import datetime

import numpy as np

from ..base.baseoptimizer import BaseOptimizer
from ..base.utils import levy


class HHO(BaseOptimizer):
    """
    Harris hawks optimization: Algorithm and applications
    """
    def __init__(self, problem, population_size, maximum_iterations, **kwargs):
        super().__init__(problem, population_size, maximum_iterations, **kwargs)

    def solve(self):
        self.solution.start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Initialize the individual positions
        for i in range(self.population_size):
            self.individual_fitness[i] = self.problem(self.individual_positions[i, :], **self.kwargs)
        for i in range(self.population_size):
            if self.individual_fitness[i] < self.solution.best_fitness:
                self.solution.best_fitness = self.individual_fitness[i]
                self.solution.best_position = self.individual_positions[i, :].copy()

        for itr in range(self.maximum_iterations):
            E1 = 2 * (1 - (itr / self.maximum_iterations))  # factor to show the decreasing energy of rabbit
            for i in range(self.population_size):
                E0 = 2 * np.random.rand() - 1  # factor to show the decreasing energy of rabbit
                escaping_energy = E1 * E0
                # phase 1: Exploration
                if np.abs(escaping_energy) >= 1:
                    q = np.random.rand()
                    rand_hawk_index = int(np.floor(self.population_size * np.random.rand()))
                    hawk_position = self.individual_positions[rand_hawk_index]
                    if q < 0.5:
                        r1 = np.random.rand()
                        r2 = np.random.rand()
                        self.individual_positions[i] = (
                                hawk_position
                                - r1
                                * np.abs(hawk_position - 2 * r2 * self.individual_positions[i])
                        )

                    else:
                        r3 = np.random.rand()
                        r4 = np.random.rand()
                        self.individual_positions[i] = (
                                self.solution.best_position - self.individual_positions.mean(axis=0)
                                - r3 * (
                                        r4 * (self.problem.upper_boundary - self.problem.lower_boundary)
                                        + self.problem.lower_boundary
                                        )
                                )
                # phase 2: Exploitation
                else:
                    r = np.random.rand()
                    if r >= 0.5 > np.abs(escaping_energy):  # Hard besiege
                        self.individual_positions[i, :] = (
                                self.solution.best_position
                                - escaping_energy * np.abs(self.solution.best_position - self.individual_positions[i, :])
                        )

                    if r >= 0.5 and np.abs(escaping_energy) >= 0.5:  # Soft besiege
                        J = 2 * (1 - np.random.rand())
                        self.individual_positions[i, :] = (
                            self.solution.best_position - self.individual_positions[i, :]
                            - escaping_energy * np.abs(J * self.solution.best_position - self.individual_positions[i, :])
                        )

                    if r < 0.5 <= np.abs(escaping_energy):  # Soft besiege with progressive rapid dives
                        J = 2 * (1 - np.random.rand())
                        Y = (
                                self.solution.best_position
                                - escaping_energy
                                * np.abs(J * self.solution.best_position - self.individual_positions[i, :])
                        )
                        Z = Y + np.multiply(np.random.randn(self.problem.dim), levy(self.problem.dim, beta=1.5))
                        fitness_Y = self.problem(Y, **self.kwargs)
                        fitness_Z = self.problem(Z, **self.kwargs)

                        self.individual_positions[i, :] = (
                            (Y if fitness_Y < fitness_Z else Z)
                            if (fitness_Y if fitness_Y < fitness_Z else fitness_Z) < self.individual_fitness[i]
                            else self.individual_positions[i, :]
                        )

                    if r < 0.5 and np.abs(escaping_energy) < 0.5:  # Hard besiege with progressive rapid dives
                        J = 2 * (1 - np.random.rand())
                        Y = (
                            self.solution.best_position
                            - escaping_energy
                            * np.abs(J * self.solution.best_position - self.individual_positions.mean(axis=0))
                        )
                        Z = Y + np.multiply(np.random.randn(self.problem.dim), levy(self.problem.dim, beta=1.5))
                        fitness_Y = self.problem(Y, **self.kwargs)
                        fitness_Z = self.problem(Z, **self.kwargs)
                        self.individual_positions[i, :] = (
                            (Y if fitness_Y < fitness_Z else Z)
                            if (fitness_Y if fitness_Y < fitness_Z else fitness_Z) < self.individual_fitness[i]
                            else self.individual_positions[i, :]
                        )


            for i in range(self.population_size):
                self.individual_fitness[i] = self.problem(self.individual_positions[i, :], **self.kwargs)
                if self.individual_fitness[i] < self.solution.best_fitness:
                    self.solution.best_fitness = self.individual_fitness[i]
                    self.solution.best_position = self.individual_positions[i, :].copy()
            self.solution.iteration_curve[itr] = self.solution.best_fitness

        return self.solution
