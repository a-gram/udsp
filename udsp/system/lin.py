"""
Linear systems

"""

from .base import System as _System
from ..signal.ndim import Signal1D as _Signal1D
from ..signal.ndim import Signal2D as _Signal2D
from ..signal.transforms import Transforms as _Transforms
from ..core import mtx as _mtx


class Filter(_System):
    """
    Abstract base class for LTSI systems

    Attributes
    ----------
    h: Signal
        A signal representing the system's impulse response

    """
    def __init__(self, h, **kwargs):
        super().__init__()
        self._h = h

    def _sysop(self):
        """
        A time-domain operation on the inputs

        For LTSI systems this function is a convolution. Since it
        is a 1-to-1 mapping it will have 1 input and 1 output.

        """
        raise NotImplementedError

    @property
    def h(self):
        return self._h.clone()


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


class FFilter(_System):
    """
    Abstract base class for frequency-space LTSI systems

    Attributes
    ----------
    h: Signal
        A signal representing the time-domain impulse response.
    hf: Signal
        The system's transfer function

    """
    def __init__(self, h, **kwargs):

        super().__init__()
        self._h = h
        self._hf = None

    def process(self, inputs=None):

        signals = []

        TO_FREQUENCY = _Transforms.FREQUENCY_DOMAIN

        # transform the input, if needed
        if type(inputs) is list:
            signals.append(
                inputs[0].transform(TO_FREQUENCY)
            )
        else:
            signals.append(
                inputs.transform(TO_FREQUENCY)
            )

        # pad and transform the IR, if needed
        if signals and self._h.domain != TO_FREQUENCY:
            self._hf = self._h.zero_pad_to(signals[0])\
                              .transform(TO_FREQUENCY)

        return super().process(signals)

    def _sysop(self):
        """
        A frequency domain operation on the inputs

        For LTSI systems this function is a product. Since it
        is a 1-to-1 mapping it will have 1 input and 1 output.

        """
        raise NotImplementedError

    @property
    def h(self):
        return self._hf.clone()


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
