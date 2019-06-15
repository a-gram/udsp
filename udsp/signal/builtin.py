"""
This module defines n-dimensional built-in signals that 
are commonly used in Digital Signal Processing.

"""

import math as _math

from .bbase import Builtin1D, Builtin2D
from ..core import mtx as _mtx
from ..core import stat as _stat
from ..core import media as _media


# ---------------------------------------------------------
#                         Mixins
# ---------------------------------------------------------


class RNGMixin:
    """
    Provides RNGs with various p.d.f.

    """
    dist = {
        "uniform": _stat.rng_uniform,
        "normal": _stat.rng_normal,
        "lorentz": _stat.rng_cauchy_lorentz,
        "laplace": _stat.rng_laplace,
    }


# ---------------------------------------------------------
#                       1D signals
# ---------------------------------------------------------


class Const1D(Builtin1D):
    """
    Constant signal

    Attributes
    ----------
    k: float
        The value of the constant

    """
    def __init__(self, k=0, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.make()

    def _generate(self, x):

        def f(n):
            return self.k

        return _mtx.vec_new(len(x), f)


class Pulse1D(Builtin1D):
    """
    Pulse (rectangular) signal

    Attributes
    ----------
    xo: float
        The central point of the pulse
    w: float
        The width of the pulse
    a: float
        The amplitude (height) of the pulse

    """
    def __init__(self,
                 xo=0,
                 w=1,
                 a=1,
                 **kwargs):

        super().__init__(**kwargs)
        self.xo = xo
        self.w = w
        self.a = a
        self.make()

    def _generate(self, x):
        x1, x2 = self.xo - self.w / 2, self.xo + self.w / 2

        def f(n):
            return self.a if x1 <= x[n] <= x2 else 0

        return _mtx.vec_new(len(x), f)


class Gaussian1D(Builtin1D):
    """
    Gaussian signal

    Attributes
    ----------
    u: float
        The mean of the Gaussian
    s: float
        The standard deviation of the Gaussian
    k: float
        The normalization factor

    """
    def __init__(self,
                 u=0,
                 s=1,
                 k=1,
                 **kwargs):

        super().__init__(**kwargs)
        self.u = u
        self.s = s
        self.k = k
        self.make()

    def _generate(self, x):

        def f(n):
            return self.k * _math.exp(
                -(x[n] - self.u) ** 2 / (2 * self.s ** 2)
            )

        return _mtx.vec_new(len(x), f)


class Logistic1D(Builtin1D):
    """
    Logistic signal

    Attributes
    ----------
    a: float
        The amplitude (maximum)
    k: float
        The steepness
    xo: float
        The central point

    """
    def __init__(self,
                 a=1,
                 k=1,
                 xo=0,
                 **kwargs):

        super().__init__(**kwargs)
        self.a = a
        self.k = k
        self.xo = xo
        self.make()

    def _generate(self, x):

        def f(n):
            return self.a / (1 + _math.exp(-self.k * (x[n] - self.xo)))

        return _mtx.vec_new(len(x), f)


class Noise1D(Builtin1D, RNGMixin):

    def __init__(self,
                 pdf="normal",
                 pdf_params=None,
                 **kwargs):
        """
        Creates a noise signal

        Parameters
        ----------
        pdf: {"uniform","normal","lorentz","laplace"}
            A str indicating the p.d.f. from which to draw the samples.
        pdf_params: None, dict
            Optional parameters accepted by the specified p.d.f.
            as key-value entries of a dictionary, as follows:

            p.d.f     parameters
            --------------------------------------------------------
            uniform   "a": <float> - The lower bound of the interval
                      "b": <float> - The upper bound of the interval
            normal    "sigma": <float> - The standard deviation
                      "trunc": (a,b) - Truncates the p.d.f. in (a,b)
            lorentz   "gamma": <float> - The scale of the p.d.f.
                      "trunc": (a,b) - Truncates the p.d.f. in (a,b)
            laplace   "lambd": <float> - The scale of the p.d.f.
                      "trunc": (a,b) - Truncates the p.d.f. in (a,b)
        kwargs: dict
            Optional arguments

        """
        super().__init__(**kwargs)
        self.pdf = pdf
        self.pdf_params = pdf_params
        self.make()

    def _generate(self, x):

        def f(n):
            return self.dist[self.pdf](**self.pdf_params or {})

        return _mtx.vec_new(len(x), f)


# ---------------------------------------------------------
#                       2D signals
# ---------------------------------------------------------


class Const2D(Builtin2D):
    """
    Constant signal

    Attributes
    ----------
    k: float
        The value of the constant

    """
    def __init__(self,
                 k=0,
                 **kwargs):

        super().__init__(**kwargs)
        self.k = k
        self.make()

    def _generate(self, x):

        def f(n, m):
            return self.k

        return _mtx.mat_new(len(x), len(x[0]), f)


class Pulse2D(Builtin2D):
    """
    Pulse (rectangular) signal

    Attributes
    ----------
    xo: tuple
        The central point of the pulse
    w: tuple
        The width of the pulse
    a: float
        The amplitude (height) of the pulse

    """
    def __init__(self,
                 xo=(0, 0),
                 w=(1, 1),
                 a=1,
                 **kwargs):

        super().__init__(**kwargs)
        self.xo = xo
        self.w = w
        self.a = a
        self.make()

    def _generate(self, x):

        x1, x2 = self.xo[0] - self.w[0] / 2, self.xo[0] + self.w[0] / 2
        y1, y2 = self.xo[1] - self.w[1] / 2, self.xo[1] + self.w[1] / 2

        def f(n, m):
            def inbox(p):
                return (y1 <= p[0] <= y2) and (x1 <= p[1] <= x2)
            return self.a if inbox(x[n][m]) else 0

        return _mtx.mat_new(len(x), len(x[0]), f)


class Gaussian2D(Builtin2D):
    """
    Gaussian signal

    Attributes
    ----------
    u: tuple
        The mean of the Gaussian
    s: tuple
        The standard deviation of the Gaussian
    k: float
        The normalization factor

    """
    def __init__(self,
                 u=(0, 0),
                 s=(1, 1),
                 k=1,
                 **kwargs):

        super().__init__(**kwargs)
        self.u = u
        self.s = s
        self.k = k
        self.make()

    def _generate(self, x):

        def f(n, m):
            return self.k * _math.exp(
                - (x[n][m][0] - self.u[0]) ** 2 / (2 * self.s[0] ** 2)
                - (x[n][m][1] - self.u[1]) ** 2 / (2 * self.s[1] ** 2)
            )

        return _mtx.mat_new(len(x), len(x[0]), f)


class Noise2D(Builtin2D, RNGMixin):

    def __init__(self,
                 pdf="normal",
                 pdf_params=None,
                 **kwargs):
        """
        Creates a noise signal

        Parameters
        ----------
        pdf: {"uniform","normal","lorentz","laplace"}
            A str indicating the p.d.f. from which to draw the samples.
        pdf_params: None, dict
            Optional parameters accepted by the specified p.d.f.
            as key-value entries of a dictionary, as follows:

            p.d.f     parameters
            --------------------------------------------------------
            uniform   "a": <float> - The lower bound of the interval
                      "b": <float> - The upper bound of the interval
            normal    "sigma": <float> - The standard deviation
                      "trunc": (a,b) - Truncates the p.d.f. in (a,b)
            lorentz   "gamma": <float> - The scale of the p.d.f.
                      "trunc": (a,b) - Truncates the p.d.f. in (a,b)
            laplace   "lambd": <float> - The scale of the p.d.f.
                      "trunc": (a,b) - Truncates the p.d.f. in (a,b)
        kwargs: dict
            Optional arguments

        """
        super().__init__(**kwargs)
        self.pdf = pdf
        self.pdf_params = pdf_params
        self.make()

    def _generate(self, x):

        def f(n, m):
            return self.dist[self.pdf](**self.pdf_params or {})

        return _mtx.mat_new(len(x), len(x[0]), f)


class GrayImage(Builtin2D):

    def __init__(self,
                 path,
                 **kwargs):

        super().__init__(**kwargs)
        self._image = _media.Image.from_file(path)
        self._length = (*reversed(self._image.metadata.size),)
        self.make()

    def _generate(self, x):

        planes = self._image.load()
        # Image is L or LA (grayscale)
        if len(planes) in (1, 2):
            yplane = planes[0]
        # Image is RGB or RGBA (colour)
        elif len(planes) in (3, 4):
            cplanes = planes if len(planes) == 3 else planes[:3]
            # Convert to grayscale
            yplane = _mtx.mat_compose(
                cplanes,
                lambda r, g, b: round(0.2126 * r +
                                      0.7152 * g +
                                      0.0722 * b)
            )
        else:
            raise RuntimeError("Bug")
        self._image = None  # we no longer need it
        return yplane
