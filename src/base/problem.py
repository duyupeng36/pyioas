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

        if not isinstance(lower_boundary, (int, float, list, tuple, np.ndarray)):
            raise TypeError("lower_boundary must be an integer, float or np.ndarray")

        if not isinstance(upper_boundary, (int, float, list, tuple, np.ndarray)):
            raise TypeError("upper_boundary must be a integer, float or np.ndarray")

        self.dim = dim





if __name__ == "__main__":
    problem = BaseProblem(-1, 0, 10)
