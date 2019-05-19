import unittest

from udsp.udsp.signal.ndim import Signal1D
from udsp.udsp.signal.plotter import Plotter1D
from udsp.udsp.signal.transforms import Transforms


class Signal1DTestCase(unittest.TestCase):

    def test_signal1d__init__(self):

        data = [1, 2, 3, 4, 5]
        datax = [0, 1, 2, 3, 4]
        name = "signal"
        xunits = "x units"
        yunits = "y units"
        signal = Signal1D(y=data,
                          xunits=xunits,
                          yunits=yunits,
                          name=name)
        self.assertEqual(signal.length, len(data))
        self.assertEqual(signal.sfreq, 1)
        self.assertEqual(signal.xunits, xunits)
        self.assertEqual(signal.yunits, yunits)
        self.assertEqual(signal.name, name)
        self.assertEqual(signal.nsamples, len(data))
        self.assertEqual(signal.dim[0], len(data))
        self.assertEqual(signal.ndim, 1)
        self.assertIsInstance(signal.plot, Plotter1D)
        y, x = signal.get(alls=True)
        self.assertListEqual(y, data)
        self.assertListEqual(x, datax)
        self.assertEqual(signal.domain, Transforms.TIMESPACE_DOMAIN)

        with self.assertRaises(ValueError):
            Signal1D(x=datax)
        with self.assertRaises(ValueError):
            Signal1D(y=data, length=-10)

    def test_signal1d__add__(self):

        s0 = Signal1D(y=[1, 2, 3, 4, 5])
        s1 = Signal1D(y=[1, 1, 1, 1, 1])
        s3 = Signal1D(y=[1, 1, 1])
        s4 = Signal1D(y=[])
        s5 = Signal1D(y=[])
        res1 = [2, 3, 4, 5, 6]
        res2 = [6, 7, 8, 9, 10]
        res3 = []
        r1 = s0 + s1
        r2 = s0 + 5
        r3 = s4 + s5
        self.assertListEqual(r1.get(), res1)
        self.assertListEqual(r2.get(), res2)
        self.assertListEqual(r3.get(), res3)
        with self.assertRaises(ValueError):
            s0 + s3

    def test_signal1d__sub__(self):

        s0 = Signal1D(y=[1, 2, 3, 4, 5])
        s1 = Signal1D(y=[1, 1, 1, 1, 1])
        s3 = Signal1D(y=[1, 1, 1])
        s4 = Signal1D(y=[])
        s5 = Signal1D(y=[])
        res1 = [0, 1, 2, 3, 4]
        res2 = [-4, -3, -2, -1, 0]
        res3 = []
        r1 = s0 - s1
        r2 = s0 - 5
        r3 = s4 - s5
        self.assertListEqual(r1.get(), res1)
        self.assertListEqual(r2.get(), res2)
        self.assertListEqual(r3.get(), res3)
        with self.assertRaises(ValueError):
            s0 - s3

    def test_signal1d__mul__(self):

        s0 = Signal1D(y=[1, 2, 3, 4, 5])
        s1 = Signal1D(y=[1, 1, 1, 1, 1])
        s3 = Signal1D(y=[1, 1, 1])
        s4 = Signal1D(y=[])
        s5 = Signal1D(y=[])
        res1 = [1, 2, 3, 4, 5]
        res2 = [5, 10, 15, 20, 25]
        res3 = []
        r1 = s0 * s1
        r2 = s0 * 5
        r3 = s4 * s5
        self.assertListEqual(r1.get(), res1)
        self.assertListEqual(r2.get(), res2)
        self.assertListEqual(r3.get(), res3)
        with self.assertRaises(ValueError):
            s0 * s3

    def test_signal1d__truediv__(self):

        s0 = Signal1D(y=[2, 4, 6, 8, 10])
        s1 = Signal1D(y=[1, 1, 1, 1, 1])
        s3 = Signal1D(y=[1, 1, 1])
        s4 = Signal1D(y=[])
        s5 = Signal1D(y=[])
        res1 = [2, 4, 6, 8, 10]
        res2 = [1, 2, 3, 4, 5]
        res3 = []
        r1 = s0 / s1
        r2 = s0 / 2
        r3 = s4 / s5
        self.assertListEqual(r1.get(), res1)
        self.assertListEqual(r2.get(), res2)
        self.assertListEqual(r3.get(), res3)
        with self.assertRaises(ValueError):
            s0 / s3

    def test_signal1d__neg__(self):

        s1 = Signal1D(y=[2, 4, 6, 8, 10])
        s2 = Signal1D(y=[1, -1, 1, -1, 1])
        s3 = Signal1D(y=[])
        res1 = [-2, -4, -6, -8, -10]
        res2 = [-1, 1, -1, 1, -1]
        res3 = []
        r1 = -s1
        r2 = -s2
        r3 = -s3
        self.assertListEqual(r1.get(), res1)
        self.assertListEqual(r2.get(), res2)
        self.assertListEqual(r3.get(), res3)

    def test_signal1d__pow__(self):

        s1 = Signal1D(y=[2, 4, 6, 8, 10])
        s2 = Signal1D(y=[4, 9, 16, 25, 36])
        s3 = Signal1D(y=[])
        res1 = [4, 16, 36, 64, 100]
        res2 = [2, 3, 4, 5, 6]
        res3 = []
        r1 = s1**2
        r2 = s2**0.5
        r3 = s3**1
        self.assertListEqual(r1.get(), res1)
        self.assertListEqual(r2.get(), res2)
        self.assertListEqual(r3.get(), res3)

    def test_signal1d_properties(self):

        data = [1, 2, 3, 4, 5]
        signal = Signal1D(y=data)
        self.assertEqual(signal.length, len(data))
        self.assertEqual(signal.sfreq, 1)
        self.assertIsInstance(signal.dim, tuple)
        self.assertEqual(len(signal.dim), 1)
        self.assertEqual(signal.dim[0], len(data))
        self.assertEqual(signal.ndim, 1)
        self.assertEqual(signal.nsamples, len(data))
        self.assertEqual(signal.domain, Transforms.TIMESPACE_DOMAIN)
        self.assertIsInstance(signal.plot, Plotter1D)
        # setters
        with self.assertRaises(AttributeError):
            signal.length = 0
        with self.assertRaises(AttributeError):
            signal.sfreq = 0
        with self.assertRaises(AttributeError):
            signal.dim = 0
        with self.assertRaises(AttributeError):
            signal.ndim = 0
        with self.assertRaises(AttributeError):
            signal.nsamples = 0
        signal.xunits = "xunits"
        signal.yunits = "yunits"
        signal.name = "signal"
        self.assertEqual(signal.xunits, "xunits")
        self.assertEqual(signal.yunits, "yunits")
        self.assertEqual(signal.name, "signal")

    def test_signal1d_set(self):

        data = [1, 2, 3, 4, 5]
        datax = [10, 11, 12, 13, 14]
        datax2 = [10, 11, 12, 13]
        signal = Signal1D()
        signal.set(data)
        self.assertListEqual(signal.get(), data)
        self.assertListEqual(signal.get(alls=True)[0], data)
        self.assertListEqual(signal.get(alls=True)[1], [0, 1, 2, 3, 4])
        self.assertEqual(signal.length, len(data))
        self.assertEqual(signal.sfreq, 1)
        signal.set(data, datax)
        self.assertListEqual(signal.get(), data)
        self.assertListEqual(signal.get(alls=True)[0], data)
        self.assertListEqual(signal.get(alls=True)[1], datax)
        with self.assertRaises(ValueError):
            signal.set([])
        with self.assertRaises(ValueError):
            signal.set([], datax)
        with self.assertRaises(ValueError):
            signal.set([], datax2)

    def test_signal1d_is_empty(self):

        signal1 = Signal1D()
        signal2 = Signal1D(y=[])
        signal3 = Signal1D(x=[])
        signal4 = Signal1D(y=[], x=[])
        self.assertTrue(signal1.is_empty())
        self.assertTrue(signal2.is_empty())
        self.assertTrue(signal3.is_empty())
        self.assertTrue(signal4.is_empty())

    def test_signal1d_clone(self):
        # TODO: Requires implementation of equality operator
        pass

    def test_signal1d_utos(self):

        data = [1, 2, 3, 4, 5]
        signal = Signal1D(y=data, length=20)
        self.assertEqual(signal.sfreq, 0.25)
        self.assertEqual(signal.length, 20)
        self.assertEqual(signal.utos(0), 0)
        self.assertEqual(signal.utos(3), 1)
        self.assertEqual(signal.utos(4), 1)
        self.assertEqual(signal.utos(6), 2)
        self.assertEqual(signal.utos(9), 2)
        self.assertEqual(signal.utos(17), 4)
        self.assertEqual(signal.utos(20), 5)

    def test_signal1d_fft(self):

        signal = Signal1D(y=[1, 2, 3, 4, 5])
        fft = signal.fft()
        self.assertEqual(fft.domain, Transforms.FREQUENCY_DOMAIN)

    def test_signal1d_ifft(self):
        signal = Signal1D(y=[1, 2, 3, 4, 5])
        fft = signal.fft()
        self.assertEqual(fft.domain, Transforms.FREQUENCY_DOMAIN)
        ifft = fft.ifft()
        self.assertEqual(ifft.domain, Transforms.TIMESPACE_DOMAIN)

    def test_signal1d_transform(self):

        signal = Signal1D(y=[1, 2, 3, 4, 5])
        t = signal.transform(Transforms.FREQUENCY_DOMAIN)
        self.assertEqual(t.domain, Transforms.FREQUENCY_DOMAIN)
        self.assertEqual(t.ndim, signal.ndim)
        self.assertEqual(t.dim, signal.dim)
        it = t.transform(Transforms.TIMESPACE_DOMAIN)
        self.assertEqual(it.domain, Transforms.TIMESPACE_DOMAIN)
        self.assertEqual(it.ndim, signal.ndim)
        self.assertEqual(it.dim, signal.dim)

    def test_signal1d_spectrum(self):

        signal = Signal1D(y=[1, 2, 3, 4, 5])
        spec = signal.fft().spectrum()
        self.assertEqual(spec.domain, Transforms.FREQUENCY_DOMAIN)
        self.assertEqual(spec.ndim, signal.ndim)
        self.assertEqual(spec.dim, signal.dim)
