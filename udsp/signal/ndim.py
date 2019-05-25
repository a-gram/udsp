"""
This module defines generic concrete implementations of the signal
abstract class for n-dimensional signals.

"""

import math as _math

from .base import Signal as Signal
from ..zombies import mtx as _mtx
from .transforms import Transforms
from .spectrums import Spectrum, Spectrum1D, Spectrum2D
from . import plotter as _plt


class Signal1D(Signal):
    """
    A specialized class for 1-dimensional signals

    """
    def __init__(self,
                 y=None,
                 x=None,
                 length=None,
                 yunits="",
                 xunits="",
                 **kwargs):

        super().__init__(**kwargs)

        if y or x:
            self.set(y, x, length)
        else:
            self._Y = []
            self._X = []
            self._length = 0
        self.yunits = yunits
        self.xunits = xunits

    def __add__(self, other):

        other = self._get_op_value(other)
        asignal = self.clone()
        asignal._Y = _mtx.vec_add(self._Y, other)
        return asignal

    def __sub__(self, other):

        other = self._get_op_value(other)
        asignal = self.clone()
        asignal._Y = _mtx.vec_sub(self._Y, other)
        return asignal

    def __mul__(self, other):

        other = self._get_op_value(other)
        asignal = self.clone()
        asignal._Y = _mtx.vec_mul(self._Y, other)
        return asignal

    def __truediv__(self, other):

        other = self._get_op_value(other)
        asignal = self.clone()
        asignal._Y = _mtx.vec_div(self._Y, other)
        return asignal

    def __neg__(self):

        asignal = self.clone()
        asignal._Y = _mtx.vec_neg(self._Y)
        return asignal

    def __pow__(self, power):

        asignal = self.clone()
        asignal._Y = _mtx.vec_pow(self._Y, power)
        return asignal

    @property
    def dim(self):
        return len(self._Y),

    @property
    def plot(self):
        return _plt.Plotter1D(self)

    def set(self, y, x=None, length=None):

        if not y and not x:
            raise ValueError(
                "At least Y must be provided."
            )

        if not y:
            raise ValueError(
                "Y can't be empty."
            )

        if x and y and not _mtx.vec_dims_equal(y, x):
            raise ValueError(
                "X and Y must be the same size."
            )

        if length is not None and length < 0:
            raise ValueError(
                "Lengths can't be negative."
            )

        if y and not x:
            x = _mtx.vec_new(len(y), lambda n: n)

        self._length = length or len(x)
        self._sfreq = len(x) / self._length
        self._X = x
        self._Y = y

    def transform(self, to_domain):

        return Transforms.get(self, to_domain)

    def spectrum(self, stype=Spectrum.POWER, scale=Spectrum.LINEAR):

        if stype not in Spectrum.TYPES:
            raise ValueError(
                "Invalid spectrum type (%s)" % stype
            )
        if scale not in Spectrum.SCALES:
            raise ValueError(
                "Invalid scale type (%s)" % scale
            )

        spec = Spectrum1D(self, scale=scale)
        spec = getattr(spec, stype)
        return spec()

    def pad(self, p, value=0):

        if self.is_empty():
            return Signal1D()

        if min(p) < 0:
            raise ValueError(
                "Negative padding values in {}".format(p)
            )

        pads = sum(p)
        dim1 = len(self._Y) + pads
        psignal = self.clone()
        psignal._Y = _mtx.vec_new(dim1, value)
        psignal._Y[p[0]: p[0] + len(self._Y)] = self._Y
        psignal._X = _mtx.vec_new(dim1, None)
        psignal._X[p[0]: p[0] + len(self._X)] = self._X
        # TODO: Shouldn't we update the signal length?
        return psignal

    def zero_pad_to(self, signal):

        dl = signal.dim[0] - self.dim[0]

        if dl < 0:
            raise ValueError("Target signal cant't be shorter")

        return self.pad((0, dl))

    def clip(self, crange):

        if self.is_empty():
            return Signal1D()

        if min(crange) < 0:
            raise ValueError(
                "Negative values in clipping range {}".format(crange)
            )

        Yc = _mtx.vec_subvec(self._Y, crange)
        Xc = _mtx.vec_subvec(self._X, crange)

        csignal = self.clone()
        csignal._Y = Yc
        csignal._X = Xc
        csignal._length = len(Yc)
        return csignal

    def flip(self, dim=None):

        Yf = _mtx.vec_reverse(self._Y)
        Xf = _mtx.vec_reverse(self._X)

        fsignal = self.clone()
        fsignal._Y = Yf
        fsignal._X = Xf
        return fsignal

    def to_real(self):

        Y = _mtx.vec_new(self.dim[0], lambda n: self._Y[n].real)
        rsignal = self.clone()
        rsignal._Y = Y
        return rsignal

    def min(self):

        return _mtx.vec_min(self._Y)

    def max(self):

        return _mtx.vec_max(self._Y)

    def energy(self):

        return _mtx.dot_product(self._Y, self._Y)

    def power(self):

        if not self.is_empty():
            return self.energy() / len(self)

    def rms(self):

        if not self.is_empty():
            return _math.sqrt(self.power())

    def mean(self):

        if not self.is_empty():
            return _mtx.vec_sum(self._Y) / len(self)

    def variance(self):

        if not self.is_empty():
            u = self.mean()
            s = _mtx.vec_sum(_mtx.vec_pow(_mtx.vec_sub(self._Y, u), 2))
            return s / len(self)

    def stddev(self):

        if not self.is_empty():
            return _math.sqrt(self.variance())

    def mse(self, signal):

        if signal.dim != self.dim:
            raise ValueError(
                "Signals must have the same dimensions"
            )
        e = _mtx.vec_sum(
              _mtx.vec_pow(_mtx.vec_sub(signal.get(), self._Y), 2)
        )
        return e / len(self)

    def rmse(self, signal):

        return _math.sqrt(self.mse(signal))

    def mae(self, signal):

        if signal.dim != self.dim:
            raise ValueError(
                "Signals must have the same dimensions"
            )
        e = _mtx.vec_sum(
              _mtx.vec_abs(_mtx.vec_sub(signal.get(), self._Y))
        )
        return e / len(self)

    def normalize(self, imin=0, imax=1):

        if imin >= imax:
            raise ValueError(
                "Invalid interval: must be imin < imax"
            )

        smin, smax = _mtx.vec_min_max(self._Y)

        if smin == smax:
            raise ValueError(
                "Can't normalize a constant signal"
            )

        k = (imax - imin) / (smax - smin)

        def fn(n):
            return (self._Y[n] - smin) * k + imin

        Yn = _mtx.vec_new(self.dim[0], fn)

        nsignal = self.clone()
        nsignal._Y = Yn
        return nsignal

    def _copy_data(self, signal):

        signal._X = _mtx.vec_copy(self._X)
        signal._Y = _mtx.vec_copy(self._Y)
        return signal


class Signal2D(Signal):
    """
    A specialized class for 2-dimensional signals

    """

    def __init__(self,
                 y=None,
                 x=None,
                 length=None,
                 yunits=("", ""),
                 xunits=("", ""),
                 **kwargs):

        super().__init__(**kwargs)

        if y or x:
            self.set(y, x, length)
        else:
            self._Y = []
            self._X = []
            self._length = (0, 0)
        self.yunits = yunits
        self.xunits = xunits

    def __add__(self, other):

        other = self._get_op_value(other)
        asignal = self.clone()
        asignal._Y = _mtx.mat_add(self._Y, other)
        return asignal

    def __sub__(self, other):

        other = self._get_op_value(other)
        asignal = self.clone()
        asignal._Y = _mtx.mat_sub(self._Y, other)
        return asignal

    def __mul__(self, other):

        other = self._get_op_value(other)
        asignal = self.clone()
        asignal._Y = _mtx.mat_mul(self._Y, other)
        return asignal

    def __truediv__(self, other):

        other = self._get_op_value(other)
        asignal = self.clone()
        asignal._Y = _mtx.mat_div(self._Y, other)
        return asignal

    def __neg__(self):

        asignal = self.clone()
        asignal._Y = _mtx.mat_neg(self._Y)
        return asignal

    def __pow__(self, power):

        asignal = self.clone()
        asignal._Y = _mtx.mat_pow(self._Y, power)
        return asignal

    @property
    def dim(self):
        if self.is_empty():
            return 0, 0
        return len(self._Y), len(self._Y[0])

    @property
    def plot(self):
        return _plt.Plotter2D(self)

    def set(self, y, x=None, length=None):

        if not y and not x:
            raise ValueError(
                "At least Y must be provided."
            )

        if not y:
            raise ValueError(
                "Y can't be empty."
            )

        if x and y and not _mtx.mat_dims_equal(y, x):
            raise ValueError(
                "X and Y must be the same size."
            )

        if length is not None:
            if min(length) < 0:
                raise ValueError("Lengths can't be negative.")
            if not all(length):
                raise ValueError("All lengths must be > 0")

        Ny, My = len(y), len(y[0])

        if y and not x:
            x = _mtx.mat_new(Ny, My, lambda n, m: (n, m))

        self._length = length or (Ny, My)
        self._sfreq = Ny / self._length[0]
        self._X = x
        self._Y = y

    def transform(self, to_domain):

        return Transforms.get(self, to_domain, 2)

    def spectrum(self, stype=Spectrum.POWER, scale=Spectrum.LINEAR):

        if stype not in Spectrum.TYPES:
            raise ValueError(
                "Invalid spectrum type (%s)" % stype
            )
        if scale not in Spectrum.SCALES:
            raise ValueError(
                "Invalid scale type (%s)" % scale
            )

        spec = Spectrum2D(self, scale=scale)
        spec = getattr(spec, stype)
        return spec()

    def pad(self, p, value=0):

        if self.is_empty():
            return Signal2D()

        pt, pb = p[0]
        pl, pr = p[1]

        Yp = _mtx.mat_pad(self._Y, (pt, pb, pl, pr), value)
        Xp = _mtx.mat_pad(self._X, (pt, pb, pl, pr), value)

        # Solution 1
        # ----------
        # if min(p[0]) < 0 or min(p[1]) < 0:
        #     raise ValueError(
        #         "Negative padding values: {}".format(p)
        #     )
        #
        # pads1, pads2 = sum(p[0]), sum(p[1])
        # dim1, dim2 = len(self._Y) + pads1, len(self._Y[0]) + pads2
        #
        # Yp = _mtx.mat_new(dim1, dim2, value)
        # Xp = _mtx.mat_new(dim1, dim2, None)
        #
        # for n, rows in enumerate(zip(self._Y, self._X)):
        #     Yp[p[0][0] + n][p[1][0]: dim2 - p[1][1]] = rows[0]
        #     Xp[p[0][0] + n][p[1][0]: dim2 - p[1][1]] = rows[1]

        # Solution 2
        # ----------
        # Yp = [[0  if (m<p[1][0] or m>=dim2-p[1][1]) or
        #              (n<p[0][0] or n>=dim1-p[0][1])
        #           else
        #              self._Y[n-p[0][0]][m-p[1][0]]
        #           for m in range(dim2)]
        #               for n in range(dim1)]
        #                  
        # Xp = [[None  if (m<p[1][0] or m>=dim2-p[1][1]) or
        #                 (n<p[0][0] or n>=dim1-p[0][1])
        #              else
        #                 self._X[n-p[0][0]][m-p[1][0]]
        #              for m in range(dim2)]
        #                  for n in range(dim1)]
        #
        # Solution 3
        # ----------
        # Yp = [
        #     ([value] * p[1][0]) + self.Y[n - p[0][0]][:] + ([value] * p[1][1])
        #     if p[0][0] <= n < (dim1 - p[0][1])
        #     else [value] * dim2
        #     for n in range(dim1)
        # ]

        psignal = self.clone()
        psignal._Y = Yp
        psignal._X = Xp
        # TODO: update the length?
        return psignal

    # TODO: maybe move the following function elsewhere
    def pad_centre_to(self, signal):

        dl1 = signal.dim[0] - self.dim[0]
        dl2 = signal.dim[1] - self.dim[1]

        if dl1 < 0 or dl2 < 0:
            raise ValueError("Target signal cant't be shorter")

        if dl1 % 2 or dl2 % 2:
            # raise ValueError("The signal can't be centre-padded")
            print("WARNING: signal can't be exactly centre-padded")

        p = ((dl1 // 2 + dl1 % 2, dl1 // 2),
             (dl2 // 2 + dl2 % 2, dl2 // 2))

        return self.pad(p)

    def zero_pad_to(self, signal):

        dl1 = signal.dim[0] - self.dim[0]
        dl2 = signal.dim[1] - self.dim[1]

        if dl1 < 0 or dl2 < 0:
            raise ValueError("Target signal cant't be shorter")

        return self.pad(((0, dl1), (0, dl2)))

    def clip(self, crange):

        if min(crange[0]) < 0 or min(crange[1]) < 0:
            raise ValueError(
                "Negative values in clipping ranges {}".format(crange)
            )

        Yc = _mtx.mat_submat(self._Y, crange)
        Xc = _mtx.mat_submat(self._X, crange)

        csignal = self.clone()
        csignal._Y = Yc
        csignal._X = Xc
        csignal._length = (len(Yc) / self._sfreq, len(Xc) / self._sfreq)
        return csignal

    def flip(self, dim=None):

        if dim and (min(dim) < 1 or max(dim)) > 2:
            raise ValueError(
                "Dimension values must be in [1,2] -> {}".format(dim)
            )

        # Flip everything if nothing is specified
        if not dim:
            dim = (1, 2)

        Yf, Xf = [], []

        # Flip along both the columns and the rows
        if 1 in dim and 2 in dim:
            Yf = _mtx.mat_reverse(self._Y)
            Xf = _mtx.mat_reverse(self._X)
        # Flip along the columns only
        elif 1 in dim:
            Yf = _mtx.mat_reverse(self._Y, rows=False)
            Xf = _mtx.mat_reverse(self._X, rows=False)
        # Flip along the rows only
        elif 2 in dim:
            Yf = _mtx.mat_reverse(self._Y, cols=False)
            Xf = _mtx.mat_reverse(self._X, cols=False)

        fsignal = self.clone()
        fsignal._Y = Yf
        fsignal._X = Xf
        return fsignal

    def to_real(self):

        Y = _mtx.mat_new(self.dim[0], self.dim[1],
                         lambda n, m: self._Y[n][m].real)
        rsignal = self.clone()
        rsignal._Y = Y
        return rsignal

    def min(self):

        return _mtx.mat_min(self._Y)

    def max(self):

        return _mtx.mat_max(self._Y)

    def energy(self):

        if not self.is_empty():
            return _mtx.mat_sum(_mtx.mat_mul(self._Y, self._Y))

    def power(self):

        if not self.is_empty():
            return self.energy() / len(self)

    def rms(self):

        if not self.is_empty():
            return _math.sqrt(self.power())

    def mean(self):

        if not self.is_empty():
            return _mtx.mat_sum(self._Y) / len(self)

    def variance(self):

        if not self.is_empty():
            u = self.mean()
            s = _mtx.mat_sum(
                  _mtx.mat_pow(_mtx.mat_sub(self._Y, u), 2)
            )
            return s / len(self)

    def stddev(self):

        if not self.is_empty():
            return _math.sqrt(self.variance())

    def mse(self, signal):

        if signal.dim != self.dim:
            raise ValueError(
                "Signals must have the same dimensions"
            )
        e = _mtx.mat_sum(
              _mtx.mat_pow(_mtx.mat_sub(signal.get(), self._Y), 2)
        )
        return e / len(self)

    def rmse(self, signal):

        return _math.sqrt(self.mse(signal))

    def mae(self, signal):

        if signal.dim != self.dim:
            raise ValueError(
                "Signals must have the same dimensions"
            )
        e = _mtx.mat_sum(
              _mtx.mat_abs(_mtx.mat_sub(signal.get(), self._Y))
        )
        return e / len(self)

    def normalize(self, imin=0, imax=1):

        if imin >= imax:
            raise ValueError(
                "Invalid interval: must be imin < imax"
            )

        smin, smax = _mtx.mat_min_max(self._Y)

        if smin == smax:
            raise ValueError(
                "Can't normalize a constant signal"
            )

        k = (imax - imin) / (smax - smin)

        def fn(n, m):
            return (self._Y[n][m] - smin) * k + imin

        Yn = _mtx.mat_new(self.dim[0], self.dim[1], fn)

        nsignal = self.clone()
        nsignal._Y = Yn
        return nsignal

    def _copy_data(self, signal):

        signal._X = _mtx.mat_copy(self._X)
        signal._Y = _mtx.mat_copy(self._Y)
        return signal
