"""
Built-in linear filters

"""

import math as _math

from .ndim import Filter1D, Filter2D
from ..signal.ndim import Signal2D as _Signal2D
from ..core import mtx as _mtx


# ---------------------------------------------------------
#                       1D filters
# ---------------------------------------------------------


class BoxFilter1D(Filter1D):
    """
    Box (average) filter

    Parameters
    ----------
    n: int
        The size of the filter. Must be odd > 1.

    """
    def __init__(self, n=5):
        super().__init__([1 / n] * n)


class TriangularFilter1D(Filter1D):
    """
    Triangular filter

    Parameters
    ----------
    n: int
        The size of the filter. Must be odd > 1.
    b: float
       Bias (offset)
    k: float
       Scale (slope)

    """
    def __init__(self, n=5, k=1, b=0):

        def fun(x):
            return b + (n - abs(x - a)) * k
        a = n // 2
        h = _mtx.vec_new(n, fun)
        h = _mtx.vec_div(h, _mtx.vec_sum(h))
        # hsum = 0
        # h = []
        # for x in range(n):
        #     hi = b + (1 - abs(x - a) / a) * k
        #     h.append(hi)
        #     hsum += hi
        # h = map(lambda x: x / hsum, h)
        super().__init__(h)


class GaussianFilter1D(Filter1D):
    """
    Gaussian filter

    Parameters
    ----------
    n: int
        The size of the filter. Must be odd > 1.
    s: float
       Standard deviation
    k: float
       Scale factor

    """
    def __init__(self, n=5, s=0.6, k=1, b=0):

        def fun(x):
            return b + _math.exp(-(x - a)**2 / (2 * s**2)) * k

        a = n // 2
        h = _mtx.vec_new(n, fun)
        h = _mtx.vec_div(h, _mtx.vec_sum(h))
        # hsum = 0
        # h = []
        # for x in range(n):
        #     hi = 2.71828**-((x - a)**2 / (2 * s**2)) * k
        #     h.append(hi)
        #     hsum += hi
        # h = map(lambda x: x / hsum, h)
        super().__init__(h)


class DiffFilter1D(Filter1D):
    """
    First order differential filter

    Parameters
    ----------
    method: str
        Derivation method (central/forward/backward)

    """
    _KERNELS = {
        "cdiff": [1, 0, -1],
        "fdiff": [1, -1, 0],
        "bdiff": [0, 1, -1]
    }

    def __init__(self, method="cdiff"):
        super().__init__(self._KERNELS[method])


class LaplacianFilter1D(Filter1D):
    """
    Second order differential filter

    """
    def __init__(self):
        super().__init__([1, -2, 1])


# ---------------------------------------------------------
#                       2D filters
# ---------------------------------------------------------


class BoxFilter2D(Filter2D):
    """
    Box (average) filter

    Parameters
    ----------
    n: int
        The size of the filer. Must be odd > 1.

    """
    def __init__(self, n=5):
        h = _mtx.mat_new(n, n, 1 / n)
        super().__init__(h)


class TriangularFilter2D(Filter2D):
    """
    Triangular filter

    Parameters
    ----------
    n: int
        The size of the filter. Must be odd > 1.
    b: float
        Bias (offset)
    k: float
        Scale (slope)
    support: str
        Kernel support ("square", "circle")

    """
    def __init__(self, n=5, k=1, b=0, support="square"):

        def pyramid(x, y):
            return b + (n - abs((x - a) + (y - a)) -
                            abs((y - a) - (x - a))) * k

        def cone(x, y):
            return b + (n - _math.sqrt((x - a)**2 +
                                       (y - a)**2)) * k

        if support not in {"square", "circle"}:
            raise ValueError(
                "Invalid support: must be 'square' or 'circle'"
            )

        fun = pyramid if support is "square" else cone

        a = n // 2
        h = _mtx.mat_new(n, n, fun)
        h = _mtx.mat_div(h, _mtx.mat_sum(h))
        super().__init__(h)


class GaussianFilter2D(Filter2D):
    """
    Gaussian filter

    Parameters
    ----------
    n: int
        The size of the filter. Must be odd > 1.
    s: float
       Standard deviation
    k: float
       Scale factor

    """
    def __init__(self, n=5, s=0.6, k=1, b=0):

        def fun(x, y):
            return b + _math.exp(-((x - a)**2 + (y - a)**2)
                                 / (2 * s**2)) * k
        a = n // 2
        h = _mtx.mat_new(n, n, fun)
        h = _mtx.mat_div(h, _mtx.mat_sum(h))
        super().__init__(h)


class DiffFilter2D(Filter2D):
    """
    First order differential filter

    Parameters
    ----------
    method: {"gradient","roberts","sobel","prewitt"}
        A string indicating the method used to compute the
        signal's derivatives. Defaults to the standard gradient
        method.

    """
    _KERNELS = {
        "gradient": {
            "hx": [[0, 0, 0],
                   [0,-1, 1],
                   [0, 0, 0]], "hy": [[0, 0, 0],
                                      [0,-1, 0],
                                      [0, 1, 0]],
        },
        "roberts": {
            "hx": [[0, 0, 0],
                   [0, 1, 0],
                   [0, 0,-1]], "hy": [[0, 0, 0],
                                      [0, 0, 1],
                                      [0,-1, 0]],
        },
        "sobel": {
            "hx": [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], "hy": [[ 1, 2, 1],
                                       [ 0, 0, 0],
                                       [-1,-2,-1]],
        },
        "prewitt": {
            "hx": [[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]], "hy": [[ 1, 1, 1],
                                       [ 0, 0, 0],
                                       [-1,-1,-1]],
        }
    }

    def __init__(self, method="gradient"):

        super().__init__(_Signal2D())
        self.method = method

    def _sysop(self):

        hx = self._KERNELS[self.method]["hx"]
        hy = self._KERNELS[self.method]["hy"]
        self._h = _Signal2D(y=hx)
        dx = super()._sysop()[0]
        self._h = _Signal2D(y=hy)
        dy = super()._sysop()[0]
        return [dx, dy]