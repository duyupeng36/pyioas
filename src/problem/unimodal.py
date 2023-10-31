"""
Unimodal benchmark functions
"""
import numpy as np

from ..base import BaseProblem


class SphereBenchmark(BaseProblem):
    """
    unimodal benchmark functions: $f(x) = \sum_{i=1}^n x_i^2$
    dim: default 30
    range: [-100, 100] for all dimensions
    minimum: 0
    """
    name = "Sphere"
    alias = "F1"

    minimum = 0

    def __init__(self, dim=30, lower_boundary=-100, upper_boundary=100, **kwargs):
        super().__init__(dim, lower_boundary, upper_boundary, **kwargs)

    def __call__(self, position, *args, **kwargs):
        return np.sum(np.square(position))


class Schwefel222Benchmark(BaseProblem):
    """
    unimodal benchmark functions: $f(x) = \sum_{i=1}^n |x_i| + \prod_{i-1}^n |x_i|$
    dim: default 30
    range: [-100, 100] for all dimensions
    minimum: 0
    """
    name = "Schwefel 2.22"
    alias = "F2"

    minimum = 0

    def __init__(self, dim=30, lower_boundary=-100, upper_boundary=100, **kwargs):
        super().__init__(dim, lower_boundary, upper_boundary, **kwargs)

    def __call__(self, position, *args, **kwargs):
        return np.sum(np.abs(position)) + np.prod(np.abs(position))




