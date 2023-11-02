"""
Fixed-dimension multimodal benchmark functions.
"""

__all__ = ['FoxholsFixedDimension', 'KowalikFixedDimension', 'SixHumpCamelFixedDimension', 'BraninFixedDimension',
           'GoldSteinPriceFixedDimension', 'Hartman3FixedDimension', 'Hartman6FixedDimension', 'Shekel5FixedDimension',
           'Shekel7FixedDimension', 'Shekel10FixedDimension']

import numpy as np
from ..base import BaseProblem


class FoxholsFixedDimension(BaseProblem):
    """
    fixed-dimension benchmark function: $f(x) = (\frac{1}{500} + \sum_{j=1}^{25}\frac{1}{j+\sum_{i=1}^{2}(x_i - a_{ij})^6})^{-1}$
    dim: 2
    range: [-65.536, 65.536] for each dimension
    minimum: 0.998
    """
    name = 'Foxhols'
    alias = "F14"

    minimum = 0.998

    def __init__(self, dim=2, lower_boundary=-65.536, upper_boundary=65.536, **kwargs):
        super(FoxholsFixedDimension, self).__init__(dim, lower_boundary, upper_boundary, **kwargs)
        self.a = np.array([[-32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 1, 6, 32],
                           [-32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 1, 6, 32]])

    def __call__(self, position, *args, **kwargs):
        b = np.zeros(25)
        for i in range(25):
            b[i] = np.sum((position - self.a[:, i]) ** 6)
        return 1 / (np.sum(1 / (np.arange(1, 26) + b)) + 0.002)


class KowalikFixedDimension(BaseProblem):
    """
    fixed-dimension benchmark function: $f(x) = \sum_{i=1}^{11}(a_i - \frac{x_1(b_i^2 + b_i x_2)}{b_i^2 + b_ix_3 + x_4})^2$
    dim: 4
    range: [-5, 5] for each dimension
    minimum: 0.0003075
    """
    name = 'Kowalik'
    alias = "F15"

    minimum = 0.0003075

    def __init__(self, dim=4, lower_boundary=-5, upper_boundary=5, **kwargs):
        super(KowalikFixedDimension, self).__init__(dim, lower_boundary, upper_boundary, **kwargs)
        self.a = np.array([0.1957, 0.1947, 0.1735, 0.16, 0.0844, 0.0627, 0.0456, 0.0342, 0.0323, 0.0235, 0.0246])
        self.b = np.array([4, 2, 1, 0.5, 0.25, 0.167, 0.125, 0.1, 0.0833, 0.0714, 0.0625])

    def __call__(self, position, *args, **kwargs):
        return np.sum((self.a - (position[0] * (self.b ** 2 + self.b * position[1])) / (self.b ** 2 + self.b * position[2] + position[3])) ** 2)


class SixHumpCamelFixedDimension(BaseProblem):
    """
    fixed-dimension benchmark function: $f(x) = 4x_1^2 - 2.1x_1^4 + \frac{x_1^6}{3} + x_1x_2 - 4x_2^2 + 4x_2^4$
    dim: 2
    range: [-5, 5] for each dimension
    minimum: -1.0316
    """

    name = 'Six-Hump Camel'
    alias = "F16"

    minimum = -1.0316

    def __init__(self, dim=2, lower_boundary=-5, upper_boundary=5, **kwargs):
        super(SixHumpCamelFixedDimension, self).__init__(dim, lower_boundary, upper_boundary, **kwargs)

    def __call__(self, position, *args, **kwargs):
        return 4 * position[0] ** 2 - 2.1 * position[0] ** 4 + position[0] ** 6 / 3 + position[0] * position[1] - 4 * position[1] ** 2 + 4 * position[1] ** 4


class BraninFixedDimension(BaseProblem):
    """
    fixed-dimension benchmark function: $f(x) =(x_2 + \frac{5.1x_1^2}{4\pi^2} _ \frac{5x_1}{\pi} - 6)^2 + 10(1 - \frac{1}{8\pi}) \cos{x_1} + 10$
    dim: 2
    range: [-5, 10] for x1, [0, 15] for x2
    minimum: 0.397887
    """

    name = 'Branin'
    alias = "F17"

    minimum = 0.397887

    def __init__(self, dim=2, lower_boundary=np.array([-5, 10]), upper_boundary=np.array([0, 15]), **kwargs):
        super(BraninFixedDimension, self).__init__(dim, lower_boundary, upper_boundary, **kwargs)

    def __call__(self, position, *args, **kwargs):
        return (position[1] - 5.1 * position[0] ** 2 / (4 * np.pi ** 2) + 5 * position[0] / np.pi - 6) ** 2 + 10 * (1 - 1 / (8 * np.pi)) * np.cos(position[0]) + 10


class GoldSteinPriceFixedDimension(BaseProblem):
    """
    fixed-dimension benchmark function: $f(x) = [1 + (x_1 + x_2 + 1)^2(19 - 14x_1 + 3x_1^2 - 14x_2 + 6x_1x_2 + 3x_2^2)][30 + (2x_1 - 3x_2)^2(18 - 32x_1 + 12x_1^2 + 48x_2 - 36x_1x_2 + 27x_2^2)]$
    dim: 2
    range: [-2, 2] for each dimension
    minimum: 3
    """

    name = 'GoldStein-Price'
    alias = "F18"

    minimum = 3

    def __init__(self, dim=2, lower_boundary=-2, upper_boundary=2, **kwargs):
        super(GoldSteinPriceFixedDimension, self).__init__(dim, lower_boundary, upper_boundary, **kwargs)

    def __call__(self, position, *args, **kwargs):
        return (1 + (position[0] + position[1] + 1) ** 2 * (19 - 14 * position[0] + 3 * position[0] ** 2 - 14 * position[1] + 6 * position[0] * position[1] + 3 * position[1] ** 2)) * (30 + (2 * position[0] - 3 * position[1]) ** 2 * (18 - 32 * position[0] + 12 * position[0] ** 2 + 48 * position[1] - 36 * position[0] * position[1] + 27 * position[1] ** 2))


class Hartman3FixedDimension(BaseProblem):
    """
    fixed-dimension benchmark function: $f(x) = -\sum_{i=1}^{4}c_i \exp(-\sum_{j=1}^{3}a_{ij}(x_j - p_{ij})^2)$
    dim: 3
    range: [0, 1] for each dimension
    minimum: -3.86278
    """

    name = 'Hartman3'
    alias = "F19"

    minimum = -3.86278

    def __init__(self, dim=3, lower_boundary=0, upper_boundary=1, **kwargs):
        super(Hartman3FixedDimension, self).__init__(dim, lower_boundary, upper_boundary, **kwargs)
        self.a = np.array([[3, 10, 30],
                           [0.1, 10, 35],
                           [3, 10, 30],
                           [0.1, 10, 35]])
        self.c = np.array([1, 1.2, 3, 3.2])
        self.p = np.array([[0.3689, 0.117, 0.2673],
                           [0.4699, 0.4387, 0.747],
                           [0.1091, 0.8732, 0.5547],
                           [0.03815, 0.5743, 0.8828]])

    def __call__(self, position, *args, **kwargs):
        return -np.sum(self.c * np.exp(-np.sum(self.a * (position - self.p) ** 2, axis=1)))


class Hartman6FixedDimension(BaseProblem):
    """
    fixed-dimension benchmark function: $f(x) = -\sum_{i=1}^{6}c_i \exp(-\sum_{j=1}^{3}a_{ij}(x_j - p_{ij})^2)$
    dim: 6
    range: [0, 1] for each dimension
    minimum: -3.32237
    """

    name = 'Hartman6'
    alias = "F20"

    minimum = -3.32237

    def __init__(self, dim=6, lower_boundary=0, upper_boundary=1, **kwargs):
        super(Hartman6FixedDimension, self).__init__(dim, lower_boundary, upper_boundary, **kwargs)
        self.a = np.array([[10, 3, 17, 3.5, 1.7, 8],
                           [0.05, 10, 17, 0.1, 8, 14],
                           [3, 3.5, 1.7, 10, 17, 8],
                           [17, 8, 0.05, 10, 0.1, 14]])
        self.c = np.array([1, 1.2, 3, 3.2])
        self.p = np.array([[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
                           [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
                           [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.665],
                           [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]])

    def __call__(self, position, *args, **kwargs):
        return -np.sum(self.c * np.exp(-np.sum(self.a * (position - self.p) ** 2, axis=1)))


class Shekel5FixedDimension(BaseProblem):
    """
    fixed-dimension benchmark function: $f(x) = -\sum_{i=1}^{5}\frac{1}{c_i + \sum_{j=1}^{4}(x_j - a_{ij})(x_j - a_{ij})^}{T}}$
    dim: 4
    range: [0, 10] for each dimension
    minimum: -10.1532
    """

    name = 'Shekel5'
    alias = "F21"

    minimum = -10.1532

    def __init__(self, dim=4, lower_boundary=0, upper_boundary=10, **kwargs):
        super(Shekel5FixedDimension, self).__init__(dim, lower_boundary, upper_boundary, **kwargs)
        self.a = np.array([[4, 4, 4, 4],
                           [1, 1, 1, 1],
                           [8, 8, 8, 8],
                           [6, 6, 6, 6],
                           [3, 7, 3, 7],
                           [2, 9, 2, 9],
                           [5, 5, 3, 3],
                           [8, 1, 8, 1],
                           [6, 2, 6, 2],
                           [7, 3.6, 7, 3.6]])
        self.c = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])

    def __call__(self, position, *args, **kwargs):
        result = 0
        for i in range(5):
            result -= 1 / (np.dot((position - self.a[i, :]), (position - self.a[i, :]).T) + self.c[i])
        return result


class Shekel7FixedDimension(BaseProblem):
    """
    fixed-dimension benchmark function: $f(x) = -\sum_{i=1}^{7}\frac{1}{c_i + \sum_{j=1}^{4}(x_j - a_{ij})(x_j - a_{ij})^}{T}}$
    dim: 4
    range: [0, 10] for each dimension
    minimum: -10.4028
    """

    name = 'Shekel7'
    alias = "F22"

    minimum = -10.4028

    def __init__(self, dim=4, lower_boundary=0, upper_boundary=10, **kwargs):
        super(Shekel7FixedDimension, self).__init__(dim, lower_boundary, upper_boundary, **kwargs)
        self.a = np.array([[4, 4, 4, 4],
                           [1, 1, 1, 1],
                           [8, 8, 8, 8],
                           [6, 6, 6, 6],
                           [3, 7, 3, 7],
                           [2, 9, 2, 9],
                           [5, 5, 3, 3],
                           [8, 1, 8, 1],
                           [6, 2, 6, 2],
                           [7, 3.6, 7, 3.6]])
        self.c = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])

    def __call__(self, position, *args, **kwargs):
        result = 0
        for i in range(7):
            result -= 1 / (np.dot((position - self.a[i, :]), (position - self.a[i, :]).T) + self.c[i])
        return result


class Shekel10FixedDimension(BaseProblem):
    """
    fixed-dimension benchmark function: $f(x) = -\sum_{i=1}^{10}\frac{1}{c_i + \sum_{j=1}^{4}(x_j - a_{ij})(x_j - a_{ij})^}{T}}$
    dim: 4
    range: [0, 10] for each dimension
    minimum: -10.5364
    """

    name = 'Shekel10'
    alias = "F23"

    minimum = -10.5364

    def __init__(self, dim=4, lower_boundary=0, upper_boundary=10, **kwargs):
        super(Shekel10FixedDimension, self).__init__(dim, lower_boundary, upper_boundary, **kwargs)
        self.a = np.array([[4, 4, 4, 4],
                           [1, 1, 1, 1],
                           [8, 8, 8, 8],
                           [6, 6, 6, 6],
                           [3, 7, 3, 7],
                           [2, 9, 2, 9],
                           [5, 5, 3, 3],
                           [8, 1, 8, 1],
                           [6, 2, 6, 2],
                           [7, 3.6, 7, 3.6]])
        self.c = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])

    def __call__(self, position, *args, **kwargs):
        result = 0
        for i in range(10):
            result -= 1 / (np.dot((position - self.a[i, :]), (position - self.a[i, :]).T) + self.c[i])
        return result
