"""
Linear systems

"""

from .fbase import Filter, FFilter
from ..signal.ndim import Signal1D as _Signal1D
from ..signal.ndim import Signal2D as _Signal2D
from ..core import mtx as _mtx


class Filter1D(Filter):
    """
    Specialized class for 1D LTSI systems

    """
    def __init__(self, h, **kwargs):

        h = _Signal1D(y=h) if type(h) is list else h
        super().__init__(h, **kwargs)

    def _sysop(self):

        x = self.inputs[0].get()
        h = self._h.get()
        a = len(h) // 2
        y = _mtx.vec_new(len(x))

        for n in range(len(x)):
            for i in range(len(h)):
                p = n + a - i
                if 0 <= p < len(x):
                    y[n] += x[p] * h[i]

        csignal = self.inputs[0].clone()
        csignal._Y = y
        return [csignal]


class Filter2D(Filter):
    """
    Specialized class for 2D LTSI systems

    """
    def __init__(self, h, **kwargs):

        h = _Signal2D(y=h) if type(h) is list else h
        super().__init__(h, **kwargs)

    def _sysop(self):

        x = self.inputs[0].get()
        h = self._h.get()
        ax = len(h[0]) // 2
        ay = len(h) // 2
        N, M = len(x), len(x[0])
        J, I = len(h), len(h[0])
        y = _mtx.mat_new(N, M)

        def inrange(pt):
            return (0 <= pt[0] < N) and (0 <= pt[1] < M)

        for n in range(N):
            for m in range(M):
                for j in range(J):
                    for i in range(I):
                        p = (n + ay - j, m + ax - i)
                        if inrange(p):
                            y[n][m] += x[p[0]][p[1]] * h[j][i]

        csignal = self.inputs[0].clone()
        csignal._Y = y
        return [csignal]


class FFilter1D(FFilter):
    """
    Specialized class for 1D frequency-space LTSI systems

    """
    def __init__(self, h, **kwargs):

        h = _Signal1D(y=h) if type(h) is list else h
        super().__init__(h, **kwargs)

    def _sysop(self):

        X = self.inputs[0].get()
        H = self._hf.get()
        Y = _mtx.vec_mul(X, H)

        csignal = self.inputs[0].clone()
        csignal._Y = Y
        return [csignal]


class FFilter2D(FFilter):
    """
    Specialized class for 2D frequency-space LTSI systems

    """
    def __init__(self, h, **kwargs):

        h = _Signal2D(y=h) if type(h) is list else h
        super().__init__(h, **kwargs)

    def _sysop(self):

        X = self.inputs[0].get()
        H = self._hf.get()
        Y = _mtx.mat_mul(X, H)

        csignal = self.inputs[0].clone()
        csignal._Y = Y
        return [csignal]
