"""
This is a base class for the problem to be optimized
"""

import numpy as np

from abc import ABCMeta, abstractmethod


class BaseProblem(metaclass=ABCMeta):

    name = "BaseProblem"

    """
    This is a base class for the problem to be optimized
    """

    def __init__(self, dim, lower_boundary, upper_boundary, **kwargs):
        """
        Parameters
        -----------
        dim : int
            the dimension of the problem

        lower_boundary : float | int | list | tuple | np.ndarray
            the lowest of the boundary of the problem

        upper_boundary : float | int | list | tuple | np.ndarray
            the upper of the boundary of the problem

        kwargs: dict
            Other parameters of the problem to be optimized
        """
        if not isinstance(dim, int) or dim < 0:
            raise ValueError("dim must be an integer and greater than 0")

        self.dim = dim

        if isinstance(lower_boundary, (int, float)):
            self.lower_boundary = np.asarray([lower_boundary] * self.dim)
        elif isinstance(lower_boundary, (list, tuple)) and len(lower_boundary) == self.dim:
            self.lower_boundary = np.asarray(lower_boundary)
        elif isinstance(lower_boundary, np.ndarray):
            self.lower_boundary = lower_boundary
        else:
            raise TypeError("lower_boundary must be an integer, float, list, tuple, np.ndarray and the length of "
                            "lower_boundary must be equal to dim")

        if isinstance(upper_boundary, (int, float)):
            self.upper_boundary = np.asarray([upper_boundary] * self.dim)
        elif isinstance(upper_boundary, (list, tuple)) and len(upper_boundary) == self.dim:
            self.upper_boundary = np.asarray(upper_boundary)
        elif isinstance(upper_boundary, np.ndarray):
            self.upper_boundary = upper_boundary
        else:
            raise TypeError("upper_boundary must be an integer, float, list, tuple, np.ndarray and the length of "
                            "upper_boundary must be equal to dim")

        self.best_position = None
        self.best_fitness = float("inf")
        self.iteration_curve = None


    @abstractmethod
    def __call__(self, position, *args, **kwargs):
        """
        abstractmethod, we must implement this method
        Parameters
        ---------

        position : np.ndarray
            the position of the problem, which means the solution of the problem
        """
        pass
