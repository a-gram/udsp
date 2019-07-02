"""
Linear filters base classes

"""

from .base import System
from ..signal.transforms import Transforms as _Transforms


class Filter(System):
    """
    Abstract base class for LTSI systems

    Attributes
    ----------
    h: Signal
        A signal representing the filter's impulse response

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


class FFilter(System):
    """
    Abstract base class for frequency-space LTSI systems

    Attributes
    ----------
    h: Signal
        A signal representing the time-domain impulse response.
    hf: Signal
        The filter's transfer function

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
