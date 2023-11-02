"""
Multimodal benchmark functions.
"""

__all__ = ['SchwefelMultimodal', 'RastriginMultimodal', 'AckleyMultimodal', 'GriewankMultimodal', 'PenalizedMultimodal',
           'Penalized2Multimodal']

import numpy as np


from ..base import BaseProblem
from ..base import u


class SchwefelMultimodal(BaseProblem):
    """
    Multimodal benchmark functions: $f(x) = \sum_{i=1}^n [-x_i \sim{(\sqrt{|x_i|})}]$
    dim: default 30
    range: [-500, 500] for all dimensions
    minimum: -418.9829 * dim
    """
    name = "Schwefel"
    alias = "F8"

    def __init__(self, dim=30, lower_boundary=-500, upper_boundary=500, **kwargs):
        super().__init__(dim, lower_boundary, upper_boundary, **kwargs)
        self.minimum = -418.9829 * self.dim

    def __call__(self, position, *args, **kwargs):
        return np.sum(-position * np.sin(np.sqrt(np.abs(position))))


class RastriginMultimodal(BaseProblem):
    """
    Multimodal benchmark functions: $f(x) = \sum_{i=1}^{n}(x_i1^2 - 10 \cos(2\pi x_i) + 10)$
    dim: default 30
    range: [-5.12, 5.12] for all dimensions
    minimum: 0
    """
    name = "Rastrigin"
    alias = "F9"

    minimum = 0

    def __init__(self, dim=30, lower_boundary=-5.12, upper_boundary=5.12, **kwargs):
        super().__init__(dim, lower_boundary, upper_boundary, **kwargs)

    def __call__(self, position, *args, **kwargs):
        return np.sum(np.square(position) - 10 * np.cos(2 * np.pi * position) + 10)


class AckleyMultimodal(BaseProblem):
    """
    Multimodal benchmark functions: $f(x) = 20 + e - 20 \exp{(-0.2 \sqrt{\frac{1}{n}\sum_{i=1}^{n} x_i^2})} - \exp{(\frac{1}{n} \sum_{i=1}^{n} \cos{(2\pi x_i)})}
    dim: default 30
    range: [-32.768, 32.768] for all dimensions
    minimum: 0
    """
    name = "Ackley"
    alias = "F10"

    minimum = 0

    def __init__(self, dim=30, lower_boundary=-32.768, upper_boundary=32.768, **kwargs):
        super().__init__(dim, lower_boundary, upper_boundary, **kwargs)

    def __call__(self, position, *args, **kwargs):
        return 20 + np.e - 20 * np.exp(-0.2 * np.sqrt(np.sum(np.square(position)) / self.dim)) - np.exp(np.sum(np.cos(2 * np.pi * position)) / self.dim)


class GriewankMultimodal(BaseProblem):
    """
    Multimodal benchmark functions: $f(x) = \frac{1}{4000}\sum_{i=1}^{n}(x_i - 100)^2 - \prod_{i=1}^{n} \cos{(\frac{x_i-100}{\sqrt{i}})} + 1$
    dim: default 30
    range: [-600, 600] for all dimensions
    minimum: 0
    """
    name = "Griewank"
    alias = "F11"

    minimum = 0

    def __init__(self, dim=30, lower_boundary=-600, upper_boundary=600, **kwargs):
        super().__init__(dim, lower_boundary, upper_boundary, **kwargs)

    def __call__(self, position, *args, **kwargs):
        return np.sum(np.square(position - 100)) / 4000 - np.prod(np.cos((position - 100) / np.sqrt(np.arange(1, self.dim + 1)))) + 1


class PenalizedMultimodal(BaseProblem):
    """
    multimodal benchmark functions: $f(x) = \frac{\pi}{n}\{10 \sin{(\pi y_1)} + \sum_{i=1}^{n-1}(y_i - 1)^2[1 + 10 \sin^2{(\pi y_{i+1})}] + (y_n-1)^2\} + \sum_{i=1}^{n}u(x_i, 10, 100, 4)$
    where $y_i = 1 + \frac{1}{4}(x_i + 1)$ $u(x_i, a, k, m) = \begin{cases} k(x_i-a)^m & x_i > a \\ 0 & -a \leq x_i \leq a \\ k(-x_i-a)^m & x_i < -a \end{cases}$
    dim: default 30
    range: [-50, 50] for all dimensions
    minimum: 0
    """

    name = "Penalized"
    alias = "F12"

    minimum = 0

    def __init__(self, dim=30, lower_boundary=-50, upper_boundary=50, **kwargs):
        super().__init__(dim, lower_boundary, upper_boundary, **kwargs)

    def __call__(self, position, *args, **kwargs):
        y = 1 + 0.25 * (position + 1)
        return np.pi / self.dim * (10 * np.sin(np.pi * y[0]) + np.sum(np.square(y[:-1] - 1) * (1 + 10 * np.square(np.sin(np.pi * y[1:])))) + np.square(y[-1] - 1)) + np.sum(u(position, 10, 100, 4))


class Penalized2Multimodal(BaseProblem):
    """
    multimodal benchmark functions: $f(x) = 0.1\{\sin^2{(3\pi x_1)} + \sum_{i=1}^{n-1}(x_i - 1)^2[1 + \sin^2{(3\pi x_i + 1)}]  + (x_n - 1)^2 \} + \sum_{i=1}^{n} u(x_i, 5, 100, 4)$
    dim: default 30
    range: [-50, 50] for all dimensions
    minimum: 0
    """
    name = "Penalized2"
    alias = "F13"

    minimum = 0

    def __init__(self, dim=30, lower_boundary=-50, upper_boundary=50, **kwargs):
        super().__init__(dim, lower_boundary, upper_boundary, **kwargs)

    def __call__(self, position, *args, **kwargs):
        return 0.1 * (np.square(np.sin(3 * np.pi * position[0])) + np.sum(np.square(position[:-1] - 1) * (1 + np.square(np.sin(3 * np.pi * position[:-1] + 1)))) + np.square(position[-1] - 1)) + np.sum(u(position, 5, 100, 4))



