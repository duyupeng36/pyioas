"""
Some utility functions
"""
import numpy as np

__all__ = ['u', "GQI"]


def u(x, a, k, m):
    """
    $u(x_i, a, k, m) = \begin{cases} k(x_i-a)^m & x_i > a \\ 0 & -a \leq x_i \leq a \\ k(-x_i-a)^m & x_i < -a \end{cases}$
    """
    return np.where(x > a, k * np.power(x - a, m), np.where(x < -a, k * np.power(-x - a, m), 0))


def interpolation(xi, xj, xk, fiti, fitj, fitk, l, u):
    """
    二次插值函数的极小值
    插值，求插值函数取得最小时的 x^*
    @return: 插值函数最小时的 x^*
    """
    a = (xj ** 2 - xk ** 2) * fiti + (xk ** 2 - xi ** 2) * fitj + (xi ** 2 - xj ** 2) * fitk
    b = 2 * ((xj - xk) * fiti + (xk - xi) * fitj + (xi - xj) * fitk)
    L_xmin = a / (b + 1e-10)
    if np.isnan(L_xmin) or np.isinf(L_xmin) or L_xmin > u or L_xmin < l:
        L_xmin = np.random.rand() * (u - l) + l
    return L_xmin


def GQI(a, b, c, fa, fb, fc, low, up):
    L_xmin = 0

    abc = np.array([fa, fb, fc])
    ind = np.argsort(abc)
    fi = abc[ind[0]]
    fj = abc[ind[1]]
    fk = abc[ind[2]]

    a_ind = ind[0]
    b_ind = ind[1]
    c_ind = ind[2]

    x = np.array([a, b, c])
    xi = x[a_ind]
    xj = x[b_ind]
    xk = x[c_ind]

    if (xk >= xi >= xj) or (xj >= xi >= xk):
        L_xmin = interpolation(xi, xj, xk, fi, fj, fk, low, up)
    elif xk >= xj >= xi:
        I = interpolation(xi, xj, xk, fi, fj, fk, low, up)
        if I < xj:
            L_xmin = I
        else:
            L_xmin = interpolation(xi, xj, 3 * xi - 2 * xj, fi, fj, fk, low, up)
    elif xi >= xj >= xk:
        I = interpolation(xi, xj, xk, fi, fj, fk, low, up)
        if I > xj:
            L_xmin = I
        else:
            L_xmin = interpolation(xi, xj, 3 * xi - 2 * xj, fi, fj, fk, low, up)
    elif xj >= xk >= xi:
        L_xmin = interpolation(xi, 2 * xi - xk, xk, fi, fj, fk, low, up)
    elif xi >= xk >= xj:
        L_xmin = interpolation(xi, 2 * xi - xk, xk, fi, fj, fk, low, up)

    return L_xmin
