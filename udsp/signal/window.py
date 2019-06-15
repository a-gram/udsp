"""
Windowing functions

"""

import math as _math

from .bbase import Builtin1D
from ..core import mtx as _mtx


class Rectangle(Builtin1D):
    """
    Rectangular window

    """
    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.make()

    def _generate(self, x):

        N = len(x)
        return _mtx.vec_new(N, lambda n: 1)


class Bartlett(Builtin1D):
    """
    Bartlett window

    """
    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.make()

    def _generate(self, x):

        N = len(x)
        return _mtx.vec_new(
            N,
            lambda n: 1 - abs((2 * n / N) - 1)
        )


class Welch(Builtin1D):
    """
    Welch window

    """
    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.make()

    def _generate(self, x):

        N = len(x)
        return _mtx.vec_new(
            N,
            lambda n: 1 - ((2 * n / N) - 1)**2
        )


class Hanning(Builtin1D):
    """
    Hanning window

    """
    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.make()

    def _generate(self, x):

        N = len(x)
        a = 2 * _math.pi / N
        return _mtx.vec_new(
            N,
            lambda n: (1 - _math.cos(a * n)) / 2
        )


class Hamming(Builtin1D):
    """
    Hamming window

    """
    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.make()

    def _generate(self, x):

        N = len(x)
        a = 2 * _math.pi / N
        return _mtx.vec_new(
            N,
            lambda n: 0.54 - 0.46 * _math.cos(a * n)
        )


class Blackman(Builtin1D):
    """
    Blackman window

    """
    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.make()

    def _generate(self, x):

        N = len(x)
        a = 2 * _math.pi / N
        return _mtx.vec_new(
            N,
            lambda n: (0.42 - 0.5 * _math.cos(a * n)
                           + 0.08 * _math.cos(2 * a * n))
        )

