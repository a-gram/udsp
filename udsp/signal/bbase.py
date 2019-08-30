"""
Base classes for builtin signals

"""

from .ndim import Signal1D
from .ndim import Signal2D
from ..core import mtx as _mtx
from ..core import utils as _utl


class Builtin1D(Signal1D):
    """
    Abstract base class for built-in 1D signals

    """
    def __init__(self, **kwargs):

        # Builtin signals automatically create the data vectors,
        # so providing them should not be allowed.
        if "y" in kwargs or "x" in kwargs:
            raise ValueError("Parameter 'y' or 'x' not allowed here")

        super().__init__(**kwargs)

    def make(self):
        """
        Creates the signal

        This method generates the signal's samples. It should be first
        called when a new instance is created, or afterwards to update
        it if parameters are changed.

        Returns
        -------
        Signal
            This signal's instance

        """
        if not self._length:
            raise ValueError(
                "The signal length must be provided"
            )

        if not self._sfreq:
            self._sfreq = 1

        dx = 1 / self._sfreq
        N = round(self._length * self._sfreq)
        X = _mtx.vec_new(N, lambda n: n * dx)
        Y = self._generate(X)

        self._X = X
        self._Y = Y
        self._length = N * dx

        print("\n"
              "Sig: {:d} samples\n"
              "Len: {} {}\n"
              "dx: {} {}\n"
              "Fs: {} samples/{}\n"
              "Dom: {:d} samples\n"
              "\n".format(
                   len(Y),
                   self._length, self.xunits,
                   dx, self.xunits,
                   self._sfreq, self.xunits,
                   len(X)))

        return self

    def _generate(self, x):
        """
        The signal's generating function

        This abstract method defines the underlying function that
        generates the signal and must be implemented by subclasses.

        Parameters
        ----------
        x: list[]
            A 1D array representing the points where the function
            must be evaluated.

        Returns
        -------
        list[]
            A 1D array with the signal's samples

        """
        raise NotImplementedError


class Builtin2D(Signal2D):
    """
    Abstract base class for built-in 2D signals

    """
    def __init__(self, **kwargs):

        # Builtin signals automatically create the data vectors,
        # so providing them should not be allowed.
        if "y" in kwargs or "x" in kwargs:
            raise ValueError("Parameter 'y' or 'x' not allowed here")

        super().__init__(**kwargs)

    def make(self):
        """
        Creates the signal

        This method generates the signal's samples. It should be first
        called when a new instance is created, or afterwards to update
        it if parameters are changed.

        Returns
        -------
        Signal
            This signal's instance

        """
        if not self._length or _utl.all_same(0, self._length):
            raise ValueError(
                "The signal length must be provided"
            )

        if not self._sfreq:
            self._sfreq = 1

        dx1 = 1 / self._sfreq
        dx2 = dx1
        dim1 = round(self._length[0] * self._sfreq)
        dim2 = round(self._length[1] * self._sfreq)

        X = _mtx.mat_new(dim1, dim2, lambda n, m: (n * dx1, m * dx2))
        Y = self._generate(X)

        assert _mtx.mat_dims_equal(X, Y, full_check=True)

        print("\n"
              "Sig: {:d}x{:d} samples\n"
              "Len: {}x{} {}\n"
              "dx:  {}x{} {}\n"
              "Fs: {} samples/{}\n"
              "Dom: {:d}x{:d} samples\n"
              "\n".format(
                   dim1, dim2,
                   self._length[0], self._length[1], self.xunits,
                   dx1, dx2, self.xunits,
                   self._sfreq, self.xunits,
                   dim1, dim2))

        self._X = X
        self._Y = Y
        self._length = (dim1 * dx1, dim2 * dx2)
        return self

    def _generate(self, x):
        """
        The signal's generating function

        This abstract method defines the underlying function that
        generates the signal and must be implemented by subclasses.

        Parameters
        ----------
        x: list[list[]]
            A 2D array representing the points where the function
            must be evaluated.

        Returns
        -------
        list[list]
            A 2D array with the signal's samples

        """
        raise NotImplementedError

