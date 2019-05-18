"""
This module defines all signal-related base classes.

"""

import copy as _copy

from .transforms import Transforms
from .spectrums import Spectrum
from ..zombies import utils as _utl


class Signal(object):
    """
    Abstract base class for signals of any kind.

    This class defines the interface that any signal must implement
    in order to provide the required DSP functionality.

    Attributes
    ----------
    _length: float, tuple
        (read-only)
        The length of the signal in each dimension in physical units.
    _sfreq: float
        (read-only)
        The sampling frequency (equal for all dimensions)
    _Y: list[]
        An n-dimensional array holding the signal's output
    _X: list[]
        An n-dimensional array holding the signal's input (the points
        at which the signal has been evaluated, if available)
    yunits: str, tuple
        The physical units for the output in each dimension.
        It is a str for 1D signals, a tuple of str otherwise
    xunits: str, tuple
        The physical units for the input in each dimension.
        It is a str for 1D signals, a tuple of str otherwise
    _domain: str
        (read-only)
        The domain the signal is currently in
    name: str
        A name assigned to the signal
    nsamples: int
        (read-only)
        The total number of samples in the signal
    dim: tuple
        (read-only)
        The per-dimension size of the signal in samples
    ndim: int
        (read-only)
        The number of dimensions of the signal
    plot: Plotter
        (read-only)
        A plotter object to display the signal

    """
    def __init__(self, sfreq=1, name=""):

        super().__init__()

        self._Y = None  # abstract
        self._X = None  # abstract
        self._length = None  # abstract
        self.xunits = None  # abstract
        self.yunits = None  # abstract
        self._sfreq = sfreq
        self._domain = Transforms.TIMESPACE_DOMAIN
        self.name = name

    # ---------------------------------------------------------
    #                       Operators
    # ---------------------------------------------------------

    def __len__(self):
        """
        Returns the total size of the signal in samples

        This overridden special method returns the total size of the signal
        by accessing the 'nsamples' property

        Returns
        -------
        int
            The total size of the signal in number of samples

        """
        return self.nsamples

    def __add__(self, other):
        raise NotImplementedError

    def __sub__(self, other):
        raise NotImplementedError

    def __mul__(self, other):
        raise NotImplementedError

    def __truediv__(self, other):
        raise NotImplementedError

    def __neg__(self):
        raise NotImplementedError

    def __pow__(self, power, modulo=None):
        raise NotImplementedError

    # ---------------------------------------------------------
    #                      Properties
    # ---------------------------------------------------------

    @property
    def length(self):
        return self._length

    @property
    def sfreq(self):
        return self._sfreq

    @property
    def domain(self):
        return self._domain

    @property
    def nsamples(self):
        return _utl.product(self.dim)

    @property
    def ndim(self):
        return len(self.dim)

    @property
    def dim(self):
        """
        Provides the per-dimension size of the signal in samples

        This method must be implemented by child classes in order to fully
        determine the size of each dimension. Not to be confused with the
        'length' attribute, which indicates the per-dimension length in
        physical units.

        Returns
        -------
        tuple
            The size of the signal in each dimension in number of samples

        """
        raise NotImplementedError

    @property
    def plot(self):
        """
        Gets a plotter for the signal

        Returns
        -------
        Plotter
            A plotter object that can be used to render the signal

        """
        raise NotImplementedError

    # ---------------------------------------------------------
    #                       Public API
    # ---------------------------------------------------------

    def get(self, alls=False):
        """
        Gets the signal's data

        Returns the signal's data, optionally with its domain of definition.

        Parameters
        ----------
        alls: bool
            Specify whether to return all the signal's data, that is the
            signal's samples (output) along with its domain (input)

        Returns
        -------
        list[], tuple[list[]]
            The signal's samples (output) or the signal's samples along
            with its domain (input) in a 2-tuple.

        """
        # TODO: should we return a copy so the data is not shared?
        if self._X and self._Y:
            return (self._Y, self._X) if alls else self._Y
        else:
            raise ValueError(
                "Signal is not properly set"
            )

    def set(self, y, x=None, length=None):
        """
        Sets the signal's data

        This abstract method must be implemented by subclasses as the data
        structures to be set depend on the nature of the signal.

        Parameters
        ----------
        y: list[]
            The signal's output data (observations)
        x: list[], optional
            The signal's 'input' data, if available (these are the points
            where the signal has been evaluated, that is its domain of
            definition). If not provided then one is created with one length
            unit per sample in all dimensions (i.e. [0,1,2,..,n])
        length: float, tuple, optional
            The length of the signal in physical units for each dimension.
            If not provided it's assumed one length unit per sample (i.e.
            equal to the number of samples)

        Returns
        -------
        Signal
            This signal

        """

        raise NotImplementedError

    def is_empty(self):
        """
        Checks whether there is signal data

        Returns
        -------
        bool

        """
        return not self._Y and not self._X

    def clone(self):
        """
        Creates a clone of the signal

        This method performs a shallow copy of the signal and then calls
        the abstract _copy_data() method that must be implemented by all
        derived classes in order to properly make a copy of the internal
        data structures.

        Returns
        -------
        Signal
            A clone of this signal

        """
        signal = _copy.copy(self)
        signal._Y = None
        signal._X = None
        self._copy_data(signal)
        return signal

    def utos(self, units):
        """
        Converts physical units to samples

        Parameters
        ----------
        units: float
            The physical units to be converted to samples

        Returns
        -------
        int
            The number of samples for the given units

        """
        return round(self.sfreq * units)

    def fft(self):
        """
        An alias for Fourier transforms

        Returns
        -------
        Signal
            The FFT of the signal

        """
        return self.transform(Transforms.FREQUENCY_DOMAIN)

    def ifft(self):
        """
        An alias for inverse Fourier transforms

        Returns
        -------
        Signal
            The IFFT of the signal

        """
        return self.transform(Transforms.TIMESPACE_DOMAIN)

    def transform(self, to_domain):
        """
        Transforms the signal to the specified domain

        This method applies a transform operation T(s) to the signal to
        transpose it from the current domain into the specified domain.

        Parameters
        ----------
        to_domain: str
            A string indicating the target domain (see Transforms.DOMAINS
            for a list of supported domains)

        Returns
        -------
        Signal
            A transformed signal

        """
        raise NotImplementedError

    def spectrum(self, stype=Spectrum.POWER, scale=Spectrum.LINEAR):
        """
        Computes the spectrum of the signal

        This method computes the spectral representation of the signal and
        is only available if the spectrum can be defined in the current
        signal's domain.

        Parameters
        ----------
        stype: {"power", "magnitude", "phase"}
            The type of spectrum
        scale: {"linear", "log"}
            The scale of the spectrum

        Returns
        -------
        Signal
            A spectral signal (scary)

        """
        raise NotImplementedError

    def pad(self, p, v=0):
        """
        Pads the signal with a given value

        Parameters
        ----------
        p: tuple
            A n-tuple of pairs (ps, pe) indicating the padding size at the
            start and at the end of the signal on each of the n dimensions
        v: scalar, optional
            A scalar value used for the padding (default is 0)

        Returns
        -------
        Signal
            A padded signal

        """
        raise NotImplementedError

    def zero_pad_to(self, signal):
        """
        Pads the signal with zeroes to match the size of another signal

        Parameters
        ----------
        signal: Signal
            A signal to whose size this signal will be padded by appending
            zeroes to it. Must have greter or equal dimensions than this
            signal.

        Returns
        -------
        Signal
            A zero-padded signal

        """
        raise NotImplementedError

    def clip(self, crange):
        """
        Extracts part of the signal

        Parameters
        ----------
        crange: tuple
            A n-tuple of pairs (cs, ce) indicating the start and end
            points of the clipped region along each dimension.

        Returns
        -------
        Signal
            A clipped signal

        """
        raise NotImplementedError

    def flip(self, dim=None):
        """
        Flips the signal along the specified dimensions

        Parameters
        ----------
        dim: tuple(int), optional
            A tuple of int values indicating the dimensions along which
            to flip the signal (e.g. for a 2D signal dim=(2,) will flip
            along the second dimension only, dim=(1, 2) will flip along
            both dimensions, etc.). If none is provided the signal will
            be flipped along all the dimensions.

        Returns
        -------
        Signal
            A flipped signal

        """
        raise NotImplementedError

    def to_real(self):
        """
        Get the real part of the signal

        Returns
        -------
        Signal
            A real signal

        """
        raise NotImplementedError

    def min(self):
        """
        Returns the minimum value of the signal

        Returns
        -------
        float

        """
        raise NotImplementedError

    def max(self):
        """
        Returns the minimum value of the signal

        Returns
        -------
        float

        """
        raise NotImplementedError

    def energy(self):
        """
        Computes the energy of the signal

        Returns
        -------
        float

        """
        raise NotImplementedError

    def power(self):
        """
        Computes the power of the signal

        Returns
        -------
        float

        """
        raise NotImplementedError

    def rms(self):
        """
        Computes the "RMS power" of the signal

        Returns
        -------
        float

        """
        raise NotImplementedError

    def mean(self):
        """
        Computes the mean of the signal

        Returns
        -------
        float

        """
        raise NotImplementedError

    def variance(self):
        """
        Computes the variance of the signal

        Returns
        -------
        float

        """
        raise NotImplementedError

    def stddev(self):
        """
        Computes the standard deviation of the signal

        Returns
        -------
        float

        """
        raise NotImplementedError

    def mse(self, signal):
        """
        Computes the Mean Square Error against a given signal

        Parameters
        ----------
        signal: Signal
            The signal against which the MSE is computed

        Returns
        -------
        float
            The MSE between the two signals

        """
        raise NotImplementedError

    def rmse(self, signal):
        """
        Computes the Root Mean Square Error against a given signal

        Parameters
        ----------
        signal: Signal
            The signal against which the RMSE is computed

        Returns
        -------
        float
            The RMSE between the two signals

        """
        raise NotImplementedError

    def mae(self, signal):
        """
        Computes the Mean Absolute Error against a given signal

        Parameters
        ----------
        signal: Signal
            The signal against which the MAE is computed

        Returns
        -------
        float
            The MAE between the two signals

        """
        raise NotImplementedError

    def norm(self, imin=0, imax=1):
        """
        Normalizes the signal in a specified interval

        Applies the generalized feature scaling to scale and
        shift the values of the signal into a specified interval.

        Parameters
        ----------
        imin: scalar
            The lower bound of the interval
        imax: scalar
            The upper bound of the interval

        Returns
        -------
        Signal
            A normalized signal

        """
        raise NotImplementedError

    # ---------------------------------------------------------
    #                      Private methods
    # ---------------------------------------------------------

    def _copy_data(self, signal):
        """
        Abstract method to delegate copy of the signal's data structures

        This method must be implemented by subclasses in order to fully
        clone the signal based on the used data structures.

        Parameters
        ----------
        signal: Signal
            A partially cloned signal to which all objects data must
            be copied.

        Returns
        -------
        Signal
            A fully cloned signal with the copied data structures

        """
        raise NotImplementedError

    def _get_op_value(self, operand):
        """
        Just a convenience refactored method

        This code is used often in operations between signals to check
        whether the operand is a signal or scalar and make the calls to
        the appropriate math functions.

        Parameters
        ----------
        operand: Signal, scalar
            A signal or a scalar value are accepted as arguments to many
            overloaded math operators. Signals must have equal sizes.

        Returns
        -------
        Signal, scalar

        """
        if isinstance(operand, Signal):
            if operand.dim != self.dim:
                raise ValueError(
                    "The signals must have equal dimensions"
                )
            return operand._Y
        return operand
