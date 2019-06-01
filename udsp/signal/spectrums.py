
import math as _math
import cmath as _cm
from ..core import mtx as _mtx


class Spectrum(object):
    """
    Abstract class for different types of spectra

    Attributes
    ----------
    _signal: Signal
        The signal for which to create the spectrum
    scale: {"linear", "log"}
        The scale of the spectrum

    """
    POWER = "power"
    MAGNITUDE = "magnitude"
    PHASE = "phase"

    LINEAR = "linear"
    LOGARITHMIC = "log"

    TYPES = {POWER, MAGNITUDE, PHASE}
    SCALES = {LINEAR, LOGARITHMIC}

    def __init__(self, signal, scale=LINEAR):
        super().__init__()
        self._signal = signal
        self.scale = scale

    def power(self):
        raise NotImplementedError

    def magnitude(self):
        raise NotImplementedError

    def phase(self):
        raise NotImplementedError

    def _tphase(self, x, th=1e-9):
        return _math.atan2(x.imag if abs(x.imag) >= th else 0,
                           x.real if abs(x.real) >= th else 0)


class Spectrum1D(Spectrum):
    """
    Specialized class for 1D spectra

    """
    def __init__(self, signal, **kwargs):
        super().__init__(signal, **kwargs)

    def power(self):

        N = len(self._signal)

        s = [(abs(y) ** 2) / (N / 2)
             if self.scale == Spectrum.LINEAR else
             _math.log((abs(y) ** 2) / (N / 2))
             for y in self._signal.get()]

        return self._make(s)

    def magnitude(self):

        N = len(self._signal)

        s = [abs(y) / (N / 2)
             if self.scale == Spectrum.LINEAR else
             _math.log(abs(y) / (N / 2))
             for y in self._signal.get()]

        return self._make(s)

    def phase(self):

        s = [self._tphase(y)
             if self.scale == Spectrum.LINEAR else
             _math.log(self._tphase(y))
             for y in self._signal.get()]

        return self._make(s)

    def _make(self, s):

        N = len(s)

        # Shift the spectrum so that the DC component is at the centre, and
        # create the frequency axis (NOTE: these operations depend on whether
        # the signal length is odd or even).
        # TODO: refactor the below code
        #       computing the shifted spectrum for N even/odd only differs by
        #       a 1 added to k (refactor with k+c => c=1 if N odd, 0 if N even)
        #       and by a = in the computation of the frequencies (refactor with
        #       a comparison function f(k,N) => k<=N if N odd, k<N if N even)
        if N % 2:
            # Shift the spectrum
            s2 = [s[(k + 1 + N // 2) % N] for k in range(N)]

            # Create the frequency axis (unshifted)
            # TODO: Can be computed as the X-axis when the fft is computed?
            fk = [(k / N) * self._signal.sfreq
                  if k <= N // 2 else
                  -((N - k) / N) * self._signal.sfreq
                  for k in range(N)]

            # Shift the frequency axis
            fk = [fk[(k + 1 + N // 2) % N] for k in range(N)]
        else:
            s2 = [s[(k + N // 2) % N] for k in range(N)]

            # TODO: same as above
            fk = [(k / N) * self._signal.sfreq
                  if k < N // 2 else
                  -((N - k) / N) * self._signal.sfreq
                  for k in range(N)]

            fk = [fk[(k + N // 2) % N] for k in range(N)]

        signal = self._signal.clone()
        signal._Y = s2
        signal._X = fk
        return signal


class Spectrum2D(Spectrum):
    """
    Specialized class for 2D spectra

    """
    def __init__(self, signal, **kwargs):
        super().__init__(signal, **kwargs)

    def power(self):
        return self._make(Spectrum.POWER)

    def magnitude(self):
        return self._make(Spectrum.MAGNITUDE)

    def phase(self):
        return self._make(Spectrum.PHASE)

    def _make(self, stype):

        N, M = self._signal.dim[0], self._signal.dim[1]
        Y = self._signal.get()
        S = _mtx.mat_new(N, M)

        # Compute the specified spectrum and shift it so that it will be
        # centered (that is the DC component at (0,0) will be at (N/2,M/2)
        # Alternatively, to shift it, multiply x[n,m] by (-1)^(n+m)
        for n in range(N):
            for m in range(M):

                if stype == Spectrum.POWER:
                    s = abs(Y[n][m]) ** 2
                elif stype == Spectrum.MAGNITUDE:
                    s = abs(Y[n][m])
                elif stype == Spectrum.PHASE:
                    s = self._tphase(Y[n][m])
                else:
                    raise RuntimeError

                if self.scale == Spectrum.LOGARITHMIC:
                    s = _math.log(s)

                S[(N // 2 + n) % N][(M // 2 + m) % M] = s

        signal = self._signal.clone()
        signal._Y = S
        # TODO: signal._X = ?
        return signal
