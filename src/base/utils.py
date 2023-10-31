"""
Some utility functions
"""
import numpy as np


def u(x, a, k, m):
    """
    $u(x_i, a, k, m) = \begin{cases} k(x_i-a)^m & x_i > a \\ 0 & -a \leq x_i \leq a \\ k(-x_i-a)^m & x_i < -a \end{cases}$
    """
    return np.where(x > a, k * np.power(x - a, m), np.where(x < -a, k * np.power(-x - a, m), 0))

