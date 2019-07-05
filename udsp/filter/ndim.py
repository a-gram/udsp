"""
Linear systems

"""

from .fbase import ConvFilter, FreqFilter
from ..signal.ndim import Signal1D as _Signal1D
from ..signal.ndim import Signal2D as _Signal2D
from ..core import mtx as _mtx


class ConvFilter1D(ConvFilter):
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

        xe = self._extend_input(x, (a, a), _mtx.vec_extend)
        dn = a if xe is not x else 0

        for n in range(len(x)):
            for i in range(len(h)):
                p = n + dn + a - i
                # Ignore the missing values if the input
                # signal has not been extended
                if (self.extmode == self.BORDER_IGNORE and
                     not (0 <= p < len(x))):
                    yn = 0
                else:
                    yn = xe[p] * h[i]
                y[n] += yn

        csignal = self.inputs[0].clone()
        csignal._Y = y
        return [csignal]


class ConvFilter2D(ConvFilter):
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

        xe = self._extend_input(x, (ay, ay, ax, ax),
                                _mtx.mat_extend)

        dn, dm = (ay, ax) if xe is not x else (0, 0)

        def inrange(pt):
            return (0 <= pt[0] < N) and (0 <= pt[1] < M)

        for n in range(N):
            for m in range(M):
                for j in range(J):
                    for i in range(I):
                        p = (n + dn + ay - j,
                             m + dm + ax - i)
                        # Ignore the missing values if the input
                        # signal has not been extended
                        if (self.extmode == self.BORDER_IGNORE and
                             not inrange(p)):
                            ynm = 0
                        else:
                            ynm = xe[p[0]][p[1]] * h[j][i]
                        y[n][m] += ynm

        csignal = self.inputs[0].clone()
        csignal._Y = y
        return [csignal]


class FreqFilter1D(FreqFilter):
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


class FreqFilter2D(FreqFilter):
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
