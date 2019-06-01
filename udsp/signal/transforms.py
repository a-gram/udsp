"""
This module defines mathematical transform operators commonly
used in Digital Signal Processing.

"""

import cmath as _cm

from ..core import mtx as _mtx
from ..core import utils as _utl


class Transform(object):
    """
    Abstract class for transform operators

    This class defines the interface to be implemented by all domain
    transform operations.

    Attributes
    ----------
    ndim: {1, 2}
        Number of dimensions of the transform
    d: {-1, 1}
        Direction (1=forward, -1=inverse)

    """
    def __init__(self, ndim=1, d=1):

        super().__init__()
        self._ndim = ndim
        self._d = d

    def forward(self, signal):
        """
        Performs the forward transform of a signal

        Parameters
        ----------
        signal: Signal
            The signal to be transformed

        Returns
        -------
        Signal
            The transformed signal

        """
        raise NotImplementedError

    def inverse(self, signal):
        """
        Performs the inverse transform of a signal

        Parameters
        ----------
        signal: Signal
            The signal to be inverse-transformed

        Returns
        -------
        Signal
            The inverse-transformed signal

        """
        raise NotImplementedError

    def execute(self, signal):
        """
        A convenient method to automatically execute the transform.

        A transform is generally executed by calling either forward()
        or inverse(). However, it is possible to specify the direction
        beforehand by setting the 'd' attribute and then later call
        this method to automatically perform the specified transform.

        Parameters
        ----------
        signal: Signal
            The signal to be transformed

        Returns
        -------
        Signal
            The transformed signal

        """
        if self._d == 1:
            return self.forward(signal)
        elif self._d == -1:
            return self.inverse(signal)
        else:
            raise RuntimeError

    @property
    def ndim(self):
        return self._ndim

    @property
    def direction(self):
        return self._d


class FourierTransform(Transform):
    """
    An implementation of the Transform interface using the Fourier
    definition.

    Much of the code for the computation of the FFT has been
    adapted from the FXT Library (https://www.jjj.de/fxt)

    """
    # Reversing bin masks
    RM1 = 0x5555555555555555
    RM2 = 0x3333333333333333
    RM4 = 0x0f0f0f0f0f0f0f0f
    RM8 = 0x00ff00ff00ff00ff
    RM16 = 0x0000ffff0000ffff

    RBP_SYMM = 4
    BPI = 64

    # Swap tables
    STBL_16 = [
        [1, 2, 3, 5, 7, 11],
        [8, 4, 12, 10, 14, 13]
    ]

    STBL_32 = [
        [1, 2, 3, 5, 6, 7, 9, 11, 13, 15, 19, 23],
        [16, 8, 24, 20, 12, 28, 18, 26, 22, 30, 25, 29]
    ]

    STBL_64 = [
        [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 19,
         21, 22, 23, 25, 27, 29, 31, 35, 37, 39, 43, 47, 55],
        [32, 16, 48, 8, 40, 24, 56, 36, 20, 52, 44, 28, 60, 34,
         50, 42, 26, 58, 38, 54, 46, 62, 49, 41, 57, 53, 61, 59]
    ]

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.PERMTBL = {
             2: self.perm2,
             4: self.perm4,
             8: self.perm8,
             16: self.perm16,
             32: self.perm32,
             64: self.perm64,
        }

    def forward(self, signal):

        if self._ndim == 1:
            # Make sure the input is complex (required by the FFT)
            x = _mtx.vec_to(complex, signal.get())
            fft = self._fft1d(x)
        elif self._ndim == 2:
            # Same as above
            x = _mtx.mat_to(complex, signal.get())
            fft = self._fft2d(x)
        else:
            raise RuntimeError

        tsignal = signal.clone()
        tsignal._Y = fft
        return tsignal

    def inverse(self, signal):

        if self._ndim == 1:
            x = _mtx.vec_to(complex, signal.get())
            ifft = self._fft1d(x, -1)
        elif self._ndim == 2:
            x = _mtx.mat_to(complex, signal.get())
            ifft = self._fft2d(x, -1)
        else:
            raise RuntimeError

        tsignal = signal.clone()
        tsignal._Y = ifft
        return tsignal

    def _fft1d(self, x, d=1):
        """
        Computes the 1D Fourier Transform using FFT algorithms

        Parameters
        ----------
        x: list[complex]
            A 1D array of complex samples. The transform is computed
            in-place, so this array will be overwritten. The size
            of the array can be arbitrary. If it's a power of 2 then
            the 'standard' FFT is used for efficiency. Any other size
            is handled using the Bluestein algorithm.
        d: {1, -1}
            The direction of the transform (1=forward, -1=inverse)

        Returns
        -------
        list[complex]
            A 1D array of complex samples. These are the DFT coefficients
            for the forward transform or the timespace domain samples
            for the inverse (in the real part).

        """
        n = len(x)

        if n == 0:
            return []

        if _utl.is_pow2(n):
            return self._fft(x, d)
        else:
            return self._gfft(x, d)

    def _fft2d(self, x, d=1):
        """
        Computes the 2D Fourier Transform using FFT algorithms

        Parameters
        ----------
        x: list[list[complex]]
            A 2D array of complex samples. The transform is performed
            in-place, so this array will be overwritten. The size can
            be arbitrary.
        d: {1, -1}
            The direction of the transform (1=forward, -1=inverse)

        Returns
        -------
        list[list[complex]]
            A 2D array of complex samples. These are the DFT coefficients
            for the forward transform or the timespace domain samples
            for the inverse (in the real part).

        """
        N, M = len(x), len(x[0])

        if N == 0 or M == 0:
            return []

        fft_n = self._fft if _utl.is_pow2(N) else self._gfft
        fft_m = self._fft if _utl.is_pow2(M) else self._gfft

        X = [None] * N

        for n, row in enumerate(x):
            X[n] = fft_n(row, d)
        for m in range(M):
            col = [X[n][m] for n in range(N)]
            ft = fft_m(col, d)
            for n in range(N):
                X[n][m] = ft[n]
        return X

    def _fft(self, x, d):
        """ Complex-to-complex Radix-2 DIT FFT algorithm """
        n = len(x)
        ldn = n.bit_length() - 1
        pi = -d * _cm.pi

        self._revbin_permute(x)

        for ldm in range(1, ldn + 1):
            m = 1 << ldm
            mh = m >> 1
            phi = pi / mh
            for j in range(mh):
                w = _cm.exp(phi * j * 1j)
                for r in range(0, n, m):
                    i0 = r + j
                    i1 = i0 + mh
                    u = x[i0]
                    v = x[i1] * w
                    x[i0] = u + v
                    x[i1] = u - v
        if d < 0:
            for i in range(n):
                x[i] = x[i] / n
        return x

    def _fft_conv(self, x, y):
        # assert len(x) == len(y)
        n = len(x)
        self._fft1d(x)
        self._fft1d(y)
        for i in range(n):
            x[i] *= y[i]
        self._fft1d(x, d=-1)

    def _gfft(self, x, d):
        """ Generalized FFT (Bluestein algorithm) """
        n = len(x)
        ldnn = 1 + _utl.floor_pow2((n << 1) - 1)
        nn = 1 << ldnn
        phi = -d * 1j * _cm.pi / n
        n2 = 2 * n

        xx = [xi for xi in x] + [0] * (nn - n)
        w, wc = [0] * nn, [0] * nn
        k2 = 0

        for k in range(n):
            w[k] = _cm.exp(phi * k2)
            wc[k] = w[k].conjugate()
            xx[k] *= w[k]
            k2 += (k + k + 1)
            if k2 > n2:
                k2 -= n2

        self._fft_conv(xx, wc)

        s = -1 if n & 1 else 1
        y = [xx[i] + s * xx[i + n] for i in range(n)]
        N = n if d < 0 else 1
        return [w[i] * y[i] / N for i in range(n)]

    def _revbin_permute(self, x):
        """ Bin permuation """
        def revbin(v, ln):
            """ Reverses the n LSBs of v """
            v = ((v & self.RM1) << 1) | ((v & (~self.RM1)) >> 1)
            v = ((v & self.RM2) << 2) | ((v & (~self.RM2)) >> 2)
            v = ((v & self.RM4) << 4) | ((v & (~self.RM4)) >> 4)
            v = ((v & self.RM8) << 8) | ((v & (~self.RM8)) >> 8)
            v = ((v & self.RM16) << 16) | ((v & (~self.RM16)) >> 16)
            v = (v << 32) | (v >> 32)
            return v >> (self.BPI - ln)

        n = len(x)

        if n <= 64:
            self._revbin_permute_short(x)
            return

        ldn = _utl.floor_pow2(n)
        nh = n >> 1
        n1 = n - 1
        nx1 = nh - 2
        nx2 = n1 - nx1

        k = 0
        r = 0

        while k < (n / self.RBP_SYMM):

            if r > k:
                _utl.swap2(k, r, x)
                _utl.swap2(n1 ^ k, n1 ^ r, x)
                _utl.swap2(nx1 ^ k, nx1 ^ r, x)
                _utl.swap2(nx2 ^ k, nx2 ^ r, x)

            k += 1
            r ^= nh

            if r > k:
                _utl.swap2(k, r, x)
                _utl.swap2(n1 ^ k, n1 ^ r, x)

            k += 1
            r = revbin(k, ldn)

            if r > k:
                _utl.swap2(k, r, x)
                _utl.swap2(n1 ^ k, n1 ^ r, x)

            k += 1
            r ^= nh

            if r > k:
                _utl.swap2(k, r, x)
                _utl.swap2(nx1 ^ k, nx1 ^ r, x)

            k += 1
            r = revbin(k, ldn)

    def perm2(self, x):
        return

    def perm4(self, x):
        _utl.swap2(1, 2, x)

    def perm8(self, x):
        _utl.swap2(1, 4, x)
        _utl.swap2(3, 6, x)

    def perm16(self, x):
        for i, j in zip(self.STBL_16[0], self.STBL_16[1]):
            _utl.swap2(i, j, x)

    def perm32(self, x):
        for i, j in zip(self.STBL_32[0], self.STBL_32[1]):
            _utl.swap2(i, j, x)

    def perm64(self, x):
        for i, j in zip(self.STBL_64[0], self.STBL_64[1]):
            _utl.swap2(i, j, x)

    def _revbin_permute_short(self, x):
        """ Bin permuation for short (<=64) signals """
        n = len(x)
        self.PERMTBL[n](x)


class Transforms(object):
    """
    This class encapsulates data and methods related to transforms

    """
    # TODO: this will probably never be a real class and all its members
    #       may be moved to the module namespace

    TIMESPACE_DOMAIN = "timespace"
    FREQUENCY_DOMAIN = "frequency"

    DOMAINS = (
        TIMESPACE_DOMAIN,
        FREQUENCY_DOMAIN,
    )

    _MAPS = {

        "timespace -> frequency": FourierTransform(),
        "frequency -> timespace": FourierTransform(d=-1),
    }

    @staticmethod
    def get(signal, to_domain, ndim=1):
        """
        Finds the transform to the specified domain

        This method looks up for a suitable transform and executes it.
        It is a sort of factory method used in the Signal interface
        where the transform is invoked through helper methods.
        It is not necessary if the transform is done programmatically
        by directly using the Transform classes.

        Parameters
        ----------
        signal: Signal
            The signal to be transformed
        to_domain: str
            A string specifying the target domain
        ndim: int
            The dimension of the transform
            # TODO: this parameter may be inferred from the signal?

        Returns
        -------
        Signal
            The transformed signal

        """
        if signal.domain not in Transforms.DOMAINS:
            raise ValueError(
                "Invalid signal domain: %s" % signal.domain
            )

        if to_domain not in Transforms.DOMAINS:
            raise ValueError(
                "Invalid 'to' domain: %s" % to_domain
            )
        # TODO: the following is a no-op, or should we raise something?
        if to_domain == signal.domain:
            print("NOTICE: signal %s is already in the %s domain" %
                  (signal.name, to_domain))
            return signal
            # raise ValueError(
            #     "Signal already in %s domain" % to_domain
            # )

        T = Transforms._MAPS[signal.domain + " -> " + to_domain]
        T._ndim = ndim

        tsignal = T.execute(signal)
        tsignal._domain = to_domain
        return tsignal
