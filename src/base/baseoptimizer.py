"""
Base class for all optimization algorithms
"""

from abc import ABCMeta, abstractmethod

import numpy as np

from problem import BaseProblem
from solution import Solution


class BaseOptimizer(metaclass=ABCMeta):

    def __init__(self, problem, population_size, maximum_iterations, **kwargs):
        """
        Parameters
        -----------
        Problem : BaseProblem
            the problem to be solved, which must be an instance of the BaseProblem

        population_size : int
            the size of the position

        maximum_iterations : int
            set the maximum number of iterations
        """

        assert isinstance(population_size, int) and isinstance(maximum_iterations, int), \
            "population_size and maximum_iterations must be integers"

        if not isinstance(problem, BaseProblem):
            raise TypeError("""
            Problem must be an instance of BaseProblem: use like
            ----------------------------------------------------
            from base import BaseProblem
            class Problem(BaseProblem):
                pass
            """)

        self.problem = problem

        self.population_size = population_size
        self.maximum_iterations = maximum_iterations

        self.individual_positions = np.zeros((self.population_size, self.problem.dim))
        self.individual_fitness = np.zeros(self.population_size)
        for i in range(self.population_size):
            self.individual_positions[i, :] = np.random.rand() * (
                    self.problem.upper_boundary - self.problem.lower_boundary
            ) + self.problem.lower_boundary


        self.solution = Solution()
        self.solution.iteration_curve = np.zeros(self.maximum_iterations)

    @abstractmethod
    def solve(self):
        """
        Iterate the main loop
        """
        return self.solution

    def space_bound(self, position):
        """
        Boundary value processing

        Parameters
        -----------
        position: np.ndarray
            current position of the problem

        Returns
        -------
        np.ndarray, which is after boundary value processing
        """
        s = (position > self.problem.upper_boundary) + (position < self.problem.lower_boundary)
        res = (
                (
                        np.random.rand()
                        * (self.problem.upper_boundary - self.problem.lower_boundary)
                        + self.problem.lower_boundary
                ) * s
                + position * (~s)
        )
        return res



if __name__ == '__main__':
    class Problem(BaseProblem):

        def __call__(self, *args, **kwargs):
            pass

    opt = BaseOptimizer(problem=Problem(30, -10, 10), population_size=10.0, maximum_iterations=500)


