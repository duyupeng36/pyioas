"""
Unimodal benchmark functions
"""
import numpy as np

from ..base.problem import BaseProblem


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


class Schwefel2_22Benchmark(BaseProblem):
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


class Schwefel1_2Benchmark(BaseProblem):
    """
    unimodal benchmark functions: $f(x) = \sum_{i=1}^n (\sum_{j=1}^i x_j)^2$
    dim: default 30
    range: [-100, 100] for all dimensions
    minimum: 0
    """
    name = "Schwefel 1.2"
    alias = "F3"

    minimum = 0

    def __init__(self, dim=30, lower_boundary=-100, upper_boundary=100, **kwargs):
        super().__init__(dim, lower_boundary, upper_boundary, **kwargs)

    def __call__(self, position, *args, **kwargs):
        return np.sum(np.square(np.cumsum(position)))


class Schwefel2_21Benchmark(BaseProblem):
    """
    unimodal benchmark functions: $f(x) = \max_i{|x_i|, 1 \leq i \leq n}$
    dim: default 30
    range: [-100, 100] for all dimensions
    minimum: 0
    """
    name = "Schwefel 2.21"
    alias = "F4"

    minimum = 0
    def __init__(self, dim=30, lower_boundary=-100, upper_boundary=100, **kwargs):
        super().__init__(dim, lower_boundary, upper_boundary, **kwargs)

    def __call__(self, position, *args, **kwargs):
        return np.max(np.abs(position))


class RosenbrockBenchmark(BaseProblem):
    """
    unimodal benchmark functions: $f(x) = \sum_{i=1}^{n-1}[(100(x_{i+1}-x_i^2)^2) + (x_i -1)^2]$
    dim: default 30
    range: [-30, 30] for all dimensions
    minimum: 0
    """
    name = "Rosenbrock"
    alias = "F5"

    minimum = 0

    def __init__(self, dim=30, lower_boundary=-100, upper_boundary=100, **kwargs):
        super().__init__(dim, lower_boundary, upper_boundary, **kwargs)

    def __call__(self, position, *args, **kwargs):
        return np.sum(100 * np.square(np.square(position[1:]) - position[:-1]) + np.square(position[:-1] - 1))


class StepBenchmark(BaseProblem):
    """
    unimodal benchmark functions: $f(x) = \sum_{i=1}^{n} (x_i - 0.5)^2$
    dim: default 30
    range: [-100, 100] for all dimensions
    minimum: 0
    """
    name = "Step"
    alias = "F6"

    minimum = 0

    def __init__(self, dim=30, lower_boundary=-100, upper_boundary=100, **kwargs):
        super().__init__(dim, lower_boundary, upper_boundary, **kwargs)

    def __call__(self, position, *args, **kwargs):
        return np.sum(np.square(position - 0.5))


class QuarticBenchmark(BaseProblem):
    """
    unimodal benchmark functions: $f(x) = \sum_{i=1}^{n} i \cdot x_i^4 + random(0, 1)$
    dim: default 30
    range: [-1.28, 1.28] for all dimensions
    minimum: 0
    """
    name = "Quartic"
    alias = "F7"

    minimum = 0

    def __init__(self, dim=30, lower_boundary=-1.28, upper_boundary=1.28, **kwargs):
        super().__init__(dim, lower_boundary, upper_boundary, **kwargs)

    def __call__(self, position, *args, **kwargs):
        return np.sum(np.arange(1, self.dim + 1) * position ** 4) + np.random.rand()

