"""
Built-in convolution filters

"""

import math as _math

from .ndim import ConvFilter1D, ConvFilter2D
from ..signal.ndim import Signal2D
from ..core import mtx as _mtx


# ---------------------------------------------------------
#                       1D filters
# ---------------------------------------------------------


class BoxFilter1D(ConvFilter1D):
    """
    Box (average) filter

    Parameters
    ----------
    n: int
        The size of the filter. Must be odd > 1.

    """
    def __init__(self,
                 n=5,
                 **kwargs):

        super().__init__([1 / n] * n, **kwargs)


class TriangularFilter1D(ConvFilter1D):
    """
    Triangular filter

    Parameters
    ----------
    n: int
        The size of the filter. Must be odd > 1.
    k: float
       Scale (slope)
    b: float
       Bias (offset)

    """
    def __init__(self,
                 n=5,
                 k=1,
                 b=0,
                 **kwargs):

        def fun(x):
            return b + (n - abs(x - a)) * k
        a = n // 2
        h = _mtx.vec_new(n, fun)
        h = _mtx.vec_div(h, _mtx.vec_sum(h))
        # hsum = 0
        # h = []
        # for x in range(n):
        #     hi = b + (n - abs(x - a)) * k
        #     h.append(hi)
        #     hsum += hi
        # h = map(lambda x: x / hsum, h)
        super().__init__(h, **kwargs)


class GaussianFilter1D(ConvFilter1D):
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
    b: float
       Bias (offset)

    """
    def __init__(self,
                 n=5,
                 s=0.6,
                 k=1,
                 b=0,
                 **kwargs):

        def fun(x):
            return b + _math.exp(-(x - a)**2 / (2 * s**2)) * k

        a = n // 2
        h = _mtx.vec_new(n, fun)
        h = _mtx.vec_div(h, _mtx.vec_sum(h))
        # hsum = 0
        # h = []
        # for x in range(n):
        #     hi = b + _math.exp(-(x - a)**2 / (2 * s**2)) * k
        #     h.append(hi)
        #     hsum += hi
        # h = map(lambda x: x / hsum, h)
        super().__init__(h, **kwargs)


class DiffFilter1D(ConvFilter1D):
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

    def __init__(self,
                 method="cdiff",
                 **kwargs):

        super().__init__(self._KERNELS[method], **kwargs)


class LaplacianFilter1D(ConvFilter1D):
    """
    Second order differential filter

    """
    def __init__(self,
                 **kwargs):

        super().__init__([1, -2, 1], **kwargs)


class LoGFilter1D(ConvFilter1D):
    """
    Laplacian of Gaussian filter

    Parameters
    ----------
    n: int
        The size of the filter. Must be odd > 1.
    s: float
       Standard deviation
    k: float
       Scale factor
    b: float
       Bias (offset)

    """

    def __init__(self,
                 n=5,
                 s=0.6,
                 k=1,
                 b=0,
                 **kwargs):

        def fun(x):
            ks1 = 2 * s ** 2
            ks2 = 2 * _math.pi * s ** 6
            fac1 = ((x - a) ** 2 - ks1) / ks2
            fac2 = _math.exp(-((x - a) ** 2) / ks1)
            return b + (fac1 * fac2) * k

        def sign(num):
            return 1 if num >= 0 else -1

        a = n // 2
        h = _mtx.vec_new(n, fun)

        # Rescale the LoG so that the sum of elements is zero
        A, B = _mtx.vec_bin(h, 2, lambda e: (e < 0, abs(e)))
        ka = A / (A + B)
        kb = 1 - ka
        Ca = ka * abs(A - B)
        Cb = kb * abs(A - B)
        maxN, minN = max(A, B), min(A, B)
        maxC, minC = max(Ca, Cb), min(Ca, Cb)
        sign_maxN = 1 if maxN == A else -1

        for i, hi in enumerate(h):
            if sign(hi) == sign_maxN:
                ki = (1 - 1 / maxN * maxC)
            else:
                ki = (1 + 1 / minN * minC)
            h[i] = hi * ki

        super().__init__(h, **kwargs)


# ---------------------------------------------------------
#                       2D filters
# ---------------------------------------------------------


class BoxFilter2D(ConvFilter2D):
    """
    Box (average) filter

    Parameters
    ----------
    n: int
        The size of the filer. Must be odd > 1.

    """
    def __init__(self,
                 n=5,
                 **kwargs):

        h = _mtx.mat_new(n, n, 1 / n)
        super().__init__(h, **kwargs)


class TriangularFilter2D(ConvFilter2D):
    """
    Triangular filter

    Parameters
    ----------
    n: int
        The size of the filter. Must be odd > 1.
    k: float
        Scale (slope)
    b: float
       Bias (offset)
    support: str
        Kernel support ("square", "circle")

    """
    def __init__(self, n=5,
                       k=1,
                       b=0,
                       support="square",
                       **kwargs):

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
        super().__init__(h, **kwargs)


class GaussianFilter2D(ConvFilter2D):
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
    b: float
       Bias (offset)

    """
    def __init__(self,
                 n=5,
                 s=0.6,
                 k=1,
                 b=0,
                 **kwargs):

        def fun(x, y):
            return b + _math.exp(-((x - a)**2 + (y - a)**2)
                                 / (2 * s**2)) * k
        a = n // 2
        h = _mtx.mat_new(n, n, fun)
        h = _mtx.mat_div(h, _mtx.mat_sum(h))
        super().__init__(h, **kwargs)


class DiffFilter2D(ConvFilter2D):
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
            "hx": [[ 0, 0, 0],
                   [-1, 0, 1],
                   [ 0, 0, 0]], "hy": [[0,-1, 0],
                                       [0, 0, 0],
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

    def __init__(self,
                 method="gradient",
                 **kwargs):

        super().__init__(Signal2D(), **kwargs)
        self.method = method

    def _sysop(self):

        hx = self._KERNELS[self.method]["hx"]
        hy = self._KERNELS[self.method]["hy"]
        self._h = Signal2D(y=hx)
        dx = super()._sysop()[0]
        self._h = Signal2D(y=hy)
        dy = super()._sysop()[0]
        return [dx, dy]


class LaplacianFilter2D(ConvFilter2D):
    """
    Second order differential filter

    Parameters
    ----------
    method: {"laplace4","laplace8"}
        Derivation method

    """
    _KERNELS = {
        "laplace4": [[ 0, -1,  0],
                     [-1,  4, -1],
                     [ 0, -1,  0]],

        "laplace8": [[-1, -1, -1],
                     [-1,  8, -1],
                     [-1, -1, -1]]
    }

    def __init__(self,
                 method="laplace8",
                 **kwargs):

        super().__init__(self._KERNELS[method], **kwargs)


class LoGFilter2D(ConvFilter2D):
    """
    Laplacian of Gaussian filter

    Parameters
    ----------
    n: int
        The size of the filter. Must be odd > 1.
    s: float
       Standard deviation
    k: float
       Scale factor
    b: float
       Bias (offset)

    """
    def __init__(self,
                 n=5,
                 s=0.6,
                 k=1,
                 b=0,
                 **kwargs):

        def fun(x, y):
            ks1 = 2 * s**2
            ks2 = 2 * _math.pi * s**6
            fac1 = ((x - a)**2 + (y - a)**2 - ks1) / ks2
            fac2 = _math.exp(-((x - a)**2 + (y - a)**2) / ks1)
            return b + (fac1 * fac2) * k

        def sign(num):
            return 1 if num >= 0 else -1

        a = n // 2
        h = _mtx.mat_new(n, n, fun)

        # Rescale the LoG so that the sum of elements is zero
        A, B = _mtx.mat_bin(h, 2, lambda e: (e < 0, abs(e)))
        ka = A / (A + B)
        kb = 1 - ka
        Ca = ka * abs(A - B)
        Cb = kb * abs(A - B)
        maxN, minN = max(A, B), min(A, B)
        maxC, minC = max(Ca, Cb), min(Ca, Cb)
        sign_maxN = 1 if maxN == A else -1

        for i, row in enumerate(h):
            for j, hij in enumerate(row):
                if sign(hij) == sign_maxN:
                    kij = (1 - 1 / maxN * maxC)
                else:
                    kij = (1 + 1 / minN * minC)
                h[i][j] = hij * kij

        super().__init__(h, **kwargs)
