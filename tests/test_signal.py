import unittest

from udsp.signal.base import Signal
from udsp.signal.ndim import Signal1D, Signal2D
from udsp.signal.transforms import Transforms
from udsp.core.plotter import Plotter1D, Plotter2D


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
        self.assertEqual(signal.dim, (len(data),))
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

    def test_signal1d__len__(self):

        s0 = Signal1D(y=[1, 2, 3, 4, 5])
        s1 = Signal1D(y=[])
        self.assertEqual(len(s0), 5)
        self.assertEqual(len(s1), 0)

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
        self.assertIsInstance(r1, Signal)
        self.assertIsInstance(r2, Signal)
        self.assertIsInstance(r3, Signal)
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
        self.assertIsInstance(r1, Signal)
        self.assertIsInstance(r2, Signal)
        self.assertIsInstance(r3, Signal)
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
        self.assertIsInstance(r1, Signal)
        self.assertIsInstance(r2, Signal)
        self.assertIsInstance(r3, Signal)
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
        self.assertIsInstance(r1, Signal)
        self.assertIsInstance(r2, Signal)
        self.assertIsInstance(r3, Signal)
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
        self.assertIsInstance(r1, Signal)
        self.assertIsInstance(r2, Signal)
        self.assertIsInstance(r3, Signal)
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
        self.assertIsInstance(r1, Signal)
        self.assertIsInstance(r2, Signal)
        self.assertIsInstance(r3, Signal)
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
        signal = Signal1D()
        self.assertEqual(signal.length, 0)
        self.assertEqual(signal.dim[0], 0)
        self.assertEqual(signal.nsamples, 0)


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
        self.assertIsInstance(fft, Signal)
        self.assertEqual(fft.domain, Transforms.FREQUENCY_DOMAIN)

    def test_signal1d_ifft(self):
        signal = Signal1D(y=[1, 2, 3, 4, 5])
        fft = signal.fft()
        self.assertIsInstance(fft, Signal)
        self.assertEqual(fft.domain, Transforms.FREQUENCY_DOMAIN)
        ifft = fft.ifft()
        self.assertIsInstance(ifft, Signal)
        self.assertEqual(ifft.domain, Transforms.TIMESPACE_DOMAIN)

    def test_signal1d_transform(self):

        signal = Signal1D(y=[1, 2, 3, 4, 5])
        t = signal.transform(Transforms.FREQUENCY_DOMAIN)
        self.assertIsInstance(t, Signal)
        self.assertEqual(t.domain, Transforms.FREQUENCY_DOMAIN)
        self.assertEqual(t.ndim, signal.ndim)
        self.assertEqual(t.dim, signal.dim)
        it = t.transform(Transforms.TIMESPACE_DOMAIN)
        self.assertIsInstance(it, Signal)
        self.assertEqual(it.domain, Transforms.TIMESPACE_DOMAIN)
        self.assertEqual(it.ndim, signal.ndim)
        self.assertEqual(it.dim, signal.dim)

    def test_signal1d_spectrum(self):

        signal = Signal1D(y=[1, 2, 3, 4, 5])
        spec = signal.fft().spectrum()
        self.assertIsInstance(spec, Signal)
        self.assertEqual(spec.domain, Transforms.FREQUENCY_DOMAIN)
        self.assertEqual(spec.ndim, signal.ndim)
        self.assertEqual(spec.dim, signal.dim)

    def test_signal1d_pad(self):

        signal = Signal1D(y=[1, 2, 3, 4, 5])
        res = [[0, 1, 2, 3, 4, 5, 0, 0, 0],
               [0, 0, 1, 2, 3, 4, 5, 0, 0],
               [1, 2, 3, 4, 5, 0],
               [0, 1, 2, 3, 4, 5],
               [1, 2, 3, 4, 5]]
        pad = [(1, 3), (2, 2), (0, 1), (1, 0), (0, 0)]
        padder = [1, 2, 3, 4, 5]
        s = []
        for p in pad:
            s.append(signal.pad(p))
        for sig, res in zip(s, res):
            self.assertIsInstance(sig, Signal)
            self.assertListEqual(sig.get(), res)
        with self.assertRaises(ValueError):
            signal.pad((-2, 2))
        with self.assertRaises(ValueError):
            signal.pad((2, -2))
        with self.assertRaises(ValueError):
            signal.pad((-2, -2))

        res = [[1, 1, 2, 3, 4, 5, 1, 1, 1],
               [2, 2, 1, 2, 3, 4, 5, 2, 2],
               [1, 2, 3, 4, 5, 3],
               [4, 1, 2, 3, 4, 5],
               [1, 2, 3, 4, 5]]
        s = []
        for p, pr in zip(pad, padder):
            s.append(signal.pad(p, pr))
        for sig, res in zip(s, res):
            self.assertListEqual(sig.get(), res)

        signal = Signal1D()
        self.assertListEqual(signal.pad(pad[0]).get(), [])

    def test_signal1d_zero_pad_to(self):

        s0 = Signal1D(y=[1, 2, 3, 4, 5])
        s1 = Signal1D(y=[1, 2, 3])
        s2 = Signal1D(y=[3, 4, 5])
        r1 = s1.zero_pad_to(s0)
        r2 = s2.zero_pad_to(s1)
        self.assertIsInstance(r1, Signal)
        self.assertListEqual(r1.get(), [1, 2, 3, 0, 0])
        self.assertListEqual(r2.get(), [3, 4, 5])
        with self.assertRaises(ValueError):
            s0.zero_pad_to(s1)

    def test_signal1d_clip(self):

        signal = Signal1D(y=[1, 2, 3, 4, 5])
        res1 = [2, 3, 4]
        res2 = [1, 2, 3, 4, 5]
        res3 = [3, 4, 5]
        res4 = [1]
        res5 = []
        r1 = signal.clip((1, 3))
        r2 = signal.clip((0, 4))
        r3 = signal.clip((2, 6))
        r4 = signal.clip((0, 0))
        r5 = signal.clip((5, 5))
        self.assertIsInstance(r1, Signal)
        self.assertListEqual(r1.get(), res1)
        self.assertListEqual(r2.get(), res2)
        self.assertListEqual(r3.get(), res3)
        self.assertListEqual(r4.get(), res4)
        self.assertListEqual(r5.get(), res5)
        with self.assertRaises(ValueError):
            signal.clip((-2, 2))
        with self.assertRaises(ValueError):
            signal.clip((2, -2))
        with self.assertRaises(ValueError):
            signal.clip((-2, -2))
        signal = Signal1D()
        self.assertListEqual(signal.clip((1, 3)).get(), [])

    def test_signal1d_flip(self):

        signal = Signal1D(y=[1, 2, 3, 4, 5])
        res1 = [5, 4, 3, 2, 1]
        res2 = [5, 4, 3, 2, 1]
        r1 = signal.flip()
        r2 = signal.flip((1,))
        self.assertIsInstance(r1, Signal)
        self.assertListEqual(r1.get(), res1)
        self.assertListEqual(r2.get(), res2)
        signal = Signal1D(y=[])
        r1 = signal.flip()
        self.assertListEqual(r1.get(), [])

    def test_signal1d_to_real(self):

        s1 = Signal1D(y=[1-1j, 1+2j, -3j, 5])
        s2 = Signal1D(y=[0, -3, 1])
        s3 = Signal1D(y=[])
        res1 = [1, 1, 0, 5]
        res2 = [0, -3, 1]
        res3 = []
        r1 = s1.to_real()
        r2 = s2.to_real()
        r3 = s3.to_real()
        self.assertIsInstance(r1, Signal)
        self.assertListEqual(r1.get(), res1)
        self.assertListEqual(r2.get(), res2)
        self.assertListEqual(r3.get(), res3)

    def test_signal1d_min(self):

        s1 = Signal1D(y=[10, -3, 4, 1, -3.1])
        s2 = Signal1D(y=[3-1j, 1+2j, -3j, 5])
        s3 = Signal1D(y=[1])
        s4 = Signal1D(y=[])
        s5 = Signal1D(y=[3, 1+2j, -3j, 5])
        res1 = -3.1
        res2 = 1+2j
        res3 = 1
        r1 = s1.min()
        r2 = s2.min()
        r3 = s3.min()
        r4 = s4.min()
        self.assertEqual(r1, res1)
        self.assertEqual(r2, res2)
        self.assertEqual(r3, res3)
        self.assertIsNone(r4)
        with self.assertRaises(TypeError):
            s5.min()  # undefined

    def test_signal1d_max(self):

        s1 = Signal1D(y=[10, -3, 4, 1, -3.1])
        s2 = Signal1D(y=[3-1j, 1+2j, -3j, 5])
        s3 = Signal1D(y=[1])
        s4 = Signal1D(y=[])
        s5 = Signal1D(y=[3, 1+2j, -3j, 5])
        res1 = 10
        res2 = 5
        res3 = 1
        r1 = s1.max()
        r2 = s2.max()
        r3 = s3.max()
        r4 = s4.max()
        self.assertEqual(r1, res1)
        self.assertEqual(r2, res2)
        self.assertEqual(r3, res3)
        self.assertIsNone(r4)
        with self.assertRaises(TypeError):
            s5.max()  # undefined

    def test_signal1d_energy(self):

        s1 = Signal1D(y=[1, 2, 3, 4, -5])
        s2 = Signal1D(y=[0, 0, 0])
        s3 = Signal1D(y=[])
        self.assertEqual(s1.energy(), 55)
        self.assertEqual(s2.energy(), 0)
        self.assertIsNone(s3.energy())

    def test_signal1d_power(self):

        s1 = Signal1D(y=[1, 2, -3, 4, 5])
        s2 = Signal1D(y=[0, 0, 0])
        s3 = Signal1D(y=[])
        self.assertEqual(s1.power(), 11)
        self.assertEqual(s2.power(), 0)
        self.assertIsNone(s3.power())

    def test_signal1d_rms(self):

        s1 = Signal1D(y=[1, 2, -3, 4, 5])
        s2 = Signal1D(y=[0, 0, 0])
        s3 = Signal1D(y=[])
        self.assertAlmostEqual(s1.rms(), 3.3166, places=3)
        self.assertEqual(s2.rms(), 0)
        self.assertIsNone(s3.rms())

    def test_signal1d_mean(self):

        s1 = Signal1D(y=[1, 2, -3, 4, 5])
        s2 = Signal1D(y=[0, 0, 0])
        s3 = Signal1D(y=[])
        self.assertAlmostEqual(s1.mean(), 1.80, places=1)
        self.assertEqual(s2.mean(), 0)
        self.assertIsNone(s3.mean())

    def test_signal1d_variance(self):

        s1 = Signal1D(y=[1, 2, -3, 4, 5])
        s2 = Signal1D(y=[0, 0, 0])
        s3 = Signal1D(y=[])
        self.assertAlmostEqual(s1.variance(), 7.76, places=1)
        self.assertEqual(s2.variance(), 0)
        self.assertIsNone(s3.variance())

    def test_signal1d_stddev(self):

        s1 = Signal1D(y=[1, 2, -3, 4, 5])
        s2 = Signal1D(y=[0, 0, 0])
        s3 = Signal1D(y=[])
        self.assertAlmostEqual(s1.stddev(), 2.7856, places=3)
        self.assertEqual(s2.stddev(), 0)
        self.assertIsNone(s3.stddev())

    def test_signal1d_mse(self):
        pass

    def test_signal1d_rmse(self):
        pass

    def test_signal1d_mae(self):
        pass

    def test_signal1d_normalize(self):

        signal = Signal1D(y=[1, 2, 3, 4, 5])
        s1 = signal.normalize()
        s2 = signal.normalize(-1, 1)
        s3 = signal.normalize(10, 100)
        s4 = signal.normalize(0.5, 1.5)
        self.assertEqual(s1.get()[0], 0)
        self.assertEqual(s1.get()[4], 1)
        self.assertEqual(s2.get()[0], -1)
        self.assertEqual(s2.get()[4], 1)
        self.assertEqual(s3.get()[0], 10)
        self.assertEqual(s3.get()[4], 100)
        self.assertEqual(s4.get()[0], 0.5)
        self.assertEqual(s4.get()[4], 1.5)


# ---------------------------------------------------------
#                      2D Signal
# ---------------------------------------------------------


class Signal2DTestCase(unittest.TestCase):

    def test_signal2d__init__(self):

        data = [[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]]
        datax = [[(0, 0), (0, 1), (0, 2)],
                 [(1, 0), (1, 1), (1, 2)],
                 [(2, 0), (2, 1), (2, 2)]]
        name = "signal"
        xunits = ("x1 units", "x2 units")
        yunits = ("y1 units", "y2 units")
        signal = Signal2D(y=data,
                          xunits=xunits,
                          yunits=yunits,
                          name=name)
        self.assertEqual(signal.length, (len(data), len(data[0])))
        self.assertEqual(signal.sfreq, 1)
        self.assertEqual(signal.xunits, xunits)
        self.assertEqual(signal.yunits, yunits)
        self.assertEqual(signal.name, name)
        self.assertEqual(signal.nsamples, len(data) * len(data[0]))
        self.assertEqual(signal.dim, (len(data), len(data[0])))
        self.assertEqual(signal.ndim, 2)
        self.assertIsInstance(signal.plot, Plotter2D)
        y, x = signal.get(alls=True)
        self.assertListEqual(y, data)
        self.assertListEqual(x, datax)
        self.assertEqual(signal.domain, Transforms.TIMESPACE_DOMAIN)
        with self.assertRaises(ValueError):
            Signal2D(x=datax)
        with self.assertRaises(ValueError):
            Signal2D(y=data, length=(10, -5))

    def test_signal2d__len__(self):

        s0 = Signal2D(y=[[1, 2, 3],
                         [4, 5, 6],
                         [7, 8, 9]])
        s1 = Signal2D(y=[])
        self.assertEqual(len(s0), 9)
        self.assertEqual(len(s1), 0)

    def test_signal2d__add__(self):

        s0 = Signal2D(y=[[1, 2, 3],
                         [4, 5, 6]])
        s1 = Signal2D(y=[[1, 1, 1],
                         [1, 1, 1]])
        s2 = Signal2D(y=[[1, 1],
                         [1, 1]])
        s3 = Signal2D(y=[])
        s4 = Signal2D(y=[])
        res1 = [[2, 3, 4], [5, 6, 7]]
        res2 = [[6, 7, 8], [9, 10, 11]]
        res3 = []
        r1 = s0 + s1
        r2 = s0 + 5
        r3 = s3 + s4
        self.assertIsInstance(r1, Signal)
        self.assertIsInstance(r2, Signal)
        self.assertIsInstance(r3, Signal)
        self.assertListEqual(r1.get(), res1)
        self.assertListEqual(r2.get(), res2)
        self.assertListEqual(r3.get(), res3)
        with self.assertRaises(ValueError):
            s0 + s2

    def test_signal2d__sub__(self):

        s0 = Signal2D(y=[[1, 2, 3],
                         [4, 5, 6]])
        s1 = Signal2D(y=[[1, 1, 1],
                         [1, 1, 1]])
        s2 = Signal2D(y=[[1, 1],
                         [1, 1]])
        s3 = Signal2D(y=[])
        s4 = Signal2D(y=[])
        res1 = [[0, 1, 2], [3, 4, 5]]
        res2 = [[-4, -3, -2], [-1, 0, 1]]
        res3 = []
        r1 = s0 - s1
        r2 = s0 - 5
        r3 = s3 - s4
        self.assertIsInstance(r1, Signal)
        self.assertIsInstance(r2, Signal)
        self.assertIsInstance(r3, Signal)
        self.assertListEqual(r1.get(), res1)
        self.assertListEqual(r2.get(), res2)
        self.assertListEqual(r3.get(), res3)
        with self.assertRaises(ValueError):
            s0 - s2

    def test_signal2d__mul__(self):

        s0 = Signal2D(y=[[1, 2, 3],
                         [4, 5, 6]])
        s1 = Signal2D(y=[[1, 1, 1],
                         [1, 1, 1]])
        s2 = Signal2D(y=[[1, 1],
                         [1, 1]])
        s3 = Signal2D(y=[])
        s4 = Signal2D(y=[])
        res1 = [[1, 2, 3], [4, 5, 6]]
        res2 = [[5, 10, 15], [20, 25, 30]]
        res3 = []
        r1 = s0 * s1
        r2 = s0 * 5
        r3 = s3 * s4
        self.assertIsInstance(r1, Signal)
        self.assertIsInstance(r2, Signal)
        self.assertIsInstance(r3, Signal)
        self.assertListEqual(r1.get(), res1)
        self.assertListEqual(r2.get(), res2)
        self.assertListEqual(r3.get(), res3)
        with self.assertRaises(ValueError):
            s0 * s2

    def test_signal2d__truediv__(self):

        s0 = Signal2D(y=[[3, 6, 15],
                         [21, 27, 33]])
        s1 = Signal2D(y=[[1, 1, 1],
                         [1, 1, 1]])
        s2 = Signal2D(y=[[1, 1],
                         [1, 1]])
        s3 = Signal2D(y=[])
        s4 = Signal2D(y=[])
        res1 = [[3, 6, 15], [21, 27, 33]]
        res2 = [[1, 2, 5], [7, 9, 11]]
        res3 = []
        r1 = s0 / s1
        r2 = s0 / 3
        r3 = s3 / s4
        self.assertIsInstance(r1, Signal)
        self.assertIsInstance(r2, Signal)
        self.assertIsInstance(r3, Signal)
        self.assertListEqual(r1.get(), res1)
        self.assertListEqual(r2.get(), res2)
        self.assertListEqual(r3.get(), res3)
        with self.assertRaises(ValueError):
            s0 / s2

    def test_signal2d__neg__(self):

        s1 = Signal2D(y=[[1, 2, 3],
                         [4, 5, 6]])
        s2 = Signal2D(y=[[1, -1, 1],
                         [1, -1, 1]])
        s3 = Signal2D(y=[])
        res1 = [[-1, -2, -3], [-4, -5, -6]]
        res2 = [[-1, 1, -1], [-1, 1, -1]]
        res3 = []
        r1 = -s1
        r2 = -s2
        r3 = -s3
        self.assertIsInstance(r1, Signal)
        self.assertIsInstance(r2, Signal)
        self.assertIsInstance(r3, Signal)
        self.assertListEqual(r1.get(), res1)
        self.assertListEqual(r2.get(), res2)
        self.assertListEqual(r3.get(), res3)

    def test_signal2d__pow__(self):

        s1 = Signal2D(y=[[1, 2, 3],
                         [4, 5, 6]])
        s2 = Signal2D(y=[[4, 9, 16],
                         [25, 36, 49]])
        s3 = Signal2D(y=[])
        res1 = [[1, 4, 9], [16, 25, 36]]
        res2 = [[2, 3, 4], [5, 6, 7]]
        res3 = []
        r1 = s1**2
        r2 = s2**0.5
        r3 = s3**1
        self.assertIsInstance(r1, Signal)
        self.assertIsInstance(r2, Signal)
        self.assertIsInstance(r3, Signal)
        self.assertListEqual(r1.get(), res1)
        self.assertListEqual(r2.get(), res2)
        self.assertListEqual(r3.get(), res3)

    def test_signal2d_properties(self):

        data = [[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]]
        signal = Signal2D(y=data)
        self.assertEqual(signal.length, (len(data), len(data[0])))
        self.assertEqual(signal.sfreq, 1)
        self.assertIsInstance(signal.dim, tuple)
        self.assertEqual(len(signal.dim), 2)
        self.assertEqual(signal.dim, (len(data), len(data[0])))
        self.assertEqual(signal.ndim, 2)
        self.assertEqual(signal.nsamples, len(data) * len(data[0]))
        self.assertEqual(signal.domain, Transforms.TIMESPACE_DOMAIN)
        self.assertIsInstance(signal.plot, Plotter2D)
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
        signal.xunits = ("x1 units", "x2 units")
        signal.yunits = ("y1 units", "y2 units")
        signal.name = "signal"
        self.assertEqual(signal.xunits, ("x1 units", "x2 units"))
        self.assertEqual(signal.yunits, ("y1 units", "y2 units"))
        self.assertEqual(signal.name, "signal")
        signal = Signal2D()
        self.assertEqual(signal.length, (0, 0))
        self.assertEqual(signal.dim, (0, 0))
        self.assertEqual(signal.nsamples, 0)

    def test_signal2d_set(self):

        data = [[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]]
        datax = [[(0, 0), (0, 1), (0, 2)],
                 [(1, 0), (1, 1), (1, 2)],
                 [(2, 0), (2, 1), (2, 2)]]
        datax2 = [[(10, 10), (10, 11), (10, 12)],
                  [(11, 10), (11, 11), (11, 12)],
                  [(12, 10), (12, 11), (12, 12)]]
        datax3 = [[(0, 0), (0, 1), (0, 2)],
                  [(1, 0), (1, 1), (1, 2)]]
        signal = Signal2D()
        signal.set(data)
        self.assertListEqual(signal.get(), data)
        self.assertListEqual(signal.get(alls=True)[0], data)
        self.assertListEqual(signal.get(alls=True)[1], datax)
        self.assertEqual(signal.length, (len(data), len(data[0])))
        self.assertEqual(signal.sfreq, 1)
        signal.set(data, datax2)
        self.assertListEqual(signal.get(), data)
        self.assertListEqual(signal.get(alls=True)[0], data)
        self.assertListEqual(signal.get(alls=True)[1], datax2)
        with self.assertRaises(ValueError):
            signal.set([])
        with self.assertRaises(ValueError):
            signal.set([], datax)
        with self.assertRaises(ValueError):
            signal.set(data, datax3)

    def test_signal2d_is_empty(self):

        signal1 = Signal2D()
        signal2 = Signal2D(y=[])
        signal3 = Signal2D(x=[])
        signal4 = Signal2D(y=[], x=[])
        self.assertTrue(signal1.is_empty())
        self.assertTrue(signal2.is_empty())
        self.assertTrue(signal3.is_empty())
        self.assertTrue(signal4.is_empty())

    def test_signal2d_clone(self):
        # TODO: Requires implementation of equality operator
        pass

    def test_signal2d_utos(self):
        # Same as the 1D case
        pass

    def test_signal2d_fft(self):

        signal = Signal2D(y=[[1, 2, 3],
                             [4, 5, 6],
                             [7, 8, 9]])
        fft = signal.fft()
        self.assertIsInstance(fft, Signal)
        self.assertEqual(fft.domain, Transforms.FREQUENCY_DOMAIN)

    def test_signal2d_ifft(self):

        signal = Signal2D(y=[[1, 2, 3],
                             [4, 5, 6],
                             [7, 8, 9]])
        fft = signal.fft()
        self.assertIsInstance(fft, Signal)
        self.assertEqual(fft.domain, Transforms.FREQUENCY_DOMAIN)
        ifft = fft.ifft()
        self.assertIsInstance(ifft, Signal)
        self.assertEqual(ifft.domain, Transforms.TIMESPACE_DOMAIN)

    def test_signal2d_transform(self):

        signal = Signal2D(y=[[1, 2, 3],
                             [4, 5, 6],
                             [7, 8, 9]])
        t = signal.transform(Transforms.FREQUENCY_DOMAIN)
        self.assertIsInstance(t, Signal)
        self.assertEqual(t.domain, Transforms.FREQUENCY_DOMAIN)
        self.assertEqual(t.ndim, signal.ndim)
        self.assertEqual(t.dim, signal.dim)
        it = t.transform(Transforms.TIMESPACE_DOMAIN)
        self.assertIsInstance(it, Signal)
        self.assertEqual(it.domain, Transforms.TIMESPACE_DOMAIN)
        self.assertEqual(it.ndim, signal.ndim)
        self.assertEqual(it.dim, signal.dim)

    def test_signal2d_spectrum(self):

        signal = Signal2D(y=[[1, 2, 3],
                             [4, 5, 6],
                             [7, 8, 9]])
        spec = signal.fft().spectrum()
        self.assertIsInstance(spec, Signal)
        self.assertEqual(spec.domain, Transforms.FREQUENCY_DOMAIN)
        self.assertEqual(spec.ndim, signal.ndim)
        self.assertEqual(spec.dim, signal.dim)

    def test_signal2d_pad(self):

        signal = Signal2D(y=[[1, 2, 3],
                             [4, 5, 6],
                             [7, 8, 9]])
        res = [
            [[0, 0, 0, 0, 0, 0],
             [0, 0, 1, 2, 3, 0],
             [0, 0, 4, 5, 6, 0],
             [0, 0, 7, 8, 9, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0]],

            [[0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 2, 3, 0, 0],
             [0, 0, 4, 5, 6, 0, 0],
             [0, 0, 7, 8, 9, 0, 0],
             [0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0]],

            [[0, 1, 2, 3],
             [0, 4, 5, 6],
             [0, 7, 8, 9],
             [0, 0, 0, 0]],

            [[0, 0, 0, 0],
             [1, 2, 3, 0],
             [4, 5, 6, 0],
             [7, 8, 9, 0]],

            [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]]
        ]
        pad = [(1, 2, 2, 1),
               (2, 2, 2, 2),
               (0, 1, 1, 0),
               (1, 0, 0, 1),
               (0, 0, 0, 0)]
        padder = [1, 2, 3, 4, 5]
        s = []
        for p in pad:
            s.append(signal.pad(p))
        for sig, res in zip(s, res):
            self.assertIsInstance(sig, Signal)
            self.assertListEqual(sig.get(), res)
        with self.assertRaises(ValueError):
            signal.pad((-2, 2, 1, 1))
        with self.assertRaises(ValueError):
            signal.pad((-2, 2, 1, -1))
        with self.assertRaises(ValueError):
            signal.pad((0, 2, -1, 1))

        res = [
            [[1, 1, 1, 1, 1, 1],
             [1, 1, 1, 2, 3, 1],
             [1, 1, 4, 5, 6, 1],
             [1, 1, 7, 8, 9, 1],
             [1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1]],

            [[2, 2, 2, 2, 2, 2, 2],
             [2, 2, 2, 2, 2, 2, 2],
             [2, 2, 1, 2, 3, 2, 2],
             [2, 2, 4, 5, 6, 2, 2],
             [2, 2, 7, 8, 9, 2, 2],
             [2, 2, 2, 2, 2, 2, 2],
             [2, 2, 2, 2, 2, 2, 2]],

            [[3, 1, 2, 3],
             [3, 4, 5, 6],
             [3, 7, 8, 9],
             [3, 3, 3, 3]],

            [[4, 4, 4, 4],
             [1, 2, 3, 4],
             [4, 5, 6, 4],
             [7, 8, 9, 4]],

            [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]]
        ]
        s = []
        for p, pr in zip(pad, padder):
            s.append(signal.pad(p, pr))
        for sig, res in zip(s, res):
            self.assertListEqual(sig.get(), res)

        signal = Signal2D()
        self.assertListEqual(signal.pad(pad[0]).get(), [])

    def test_signal2d_zero_pad_to(self):

        s0 = Signal2D(y=[[1, 2, 3],
                         [4, 5, 6],
                         [7, 8, 9]])
        s1 = Signal2D(y=[[1, 2],
                         [4, 5]])
        s2 = Signal2D(y=[[2, 3],
                         [5, 6]])
        res1 = [[1, 2, 0],
                [4, 5, 0],
                [0, 0, 0]]
        res2 = [[2, 3],
                [5, 6]]
        r1 = s1.zero_pad_to(s0)
        r2 = s2.zero_pad_to(s1)
        self.assertIsInstance(r1, Signal)
        self.assertListEqual(r1.get(), res1)
        self.assertListEqual(r2.get(), res2)
        with self.assertRaises(ValueError):
            s0.zero_pad_to(s1)

    def test_signal2d_clip(self):

        signal = Signal2D(y=[[1, 2, 3],
                             [4, 5, 6],
                             [7, 8, 9]])
        res1 = [[5, 6],
                [8, 9]]
        res2 = [[2],
                [5],
                [8]]
        res3 = [[5]]
        res4 = [[7, 8, 9]]
        res5 = [[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]]
        res6 = []
        res7 = []
        res8 = [[8, 9],
                [5, 6]]
        res9 = [[9, 8, 7],
                [6, 5, 4],
                [3, 2, 1]]
        r1 = signal.clip((1, 2, 1, 2))
        r2 = signal.clip((0, 3, 1, 1))
        r3 = signal.clip((1, 1, 1, 1))
        r4 = signal.clip((2, 2, 0, 2))
        r5 = signal.clip((0, 2, 0, 2))
        r6 = signal.clip((4, 4, 3, 4))
        r7 = signal.clip((0, 2, 3, 5))
        r8 = signal.clip((2, 1, 1, 2))
        r9 = signal.clip((2, 0, 2, 0))
        self.assertIsInstance(r1, Signal)
        self.assertListEqual(r1.get(), res1)
        self.assertListEqual(r2.get(), res2)
        self.assertListEqual(r3.get(), res3)
        self.assertListEqual(r4.get(), res4)
        self.assertListEqual(r5.get(), res5)
        self.assertListEqual(r6.get(), res6)
        self.assertListEqual(r7.get(), res7)
        self.assertListEqual(r8.get(), res8)
        self.assertListEqual(r9.get(), res9)
        with self.assertRaises(ValueError):
            signal.clip((-2, 2, 1, 1))
        with self.assertRaises(ValueError):
            signal.clip((-2, 2, 1, -1))
        with self.assertRaises(ValueError):
            signal.clip((0, 2, -1, 1))
        signal = Signal2D()
        r1 = signal.clip((1, 2, 1, 2))
        self.assertListEqual(r1.get(), [])

    def test_signal2d_flip(self):

        signal = Signal2D(y=[[1, 2, 3],
                             [4, 5, 6],
                             [7, 8, 9]])
        res1 = [[9, 8, 7],
                [6, 5, 4],
                [3, 2, 1]]
        res2 = [[7, 8, 9],
                [4, 5, 6],
                [1, 2, 3]]
        res3 = [[3, 2, 1],
                [6, 5, 4],
                [9, 8, 7]]
        res4 = [[9, 8, 7],
                [6, 5, 4],
                [3, 2, 1]]
        r1 = signal.flip()
        r2 = signal.flip((1,))
        r3 = signal.flip((2,))
        r4 = signal.flip((1, 2))
        self.assertIsInstance(r1, Signal)
        self.assertListEqual(r1.get(), res1)
        self.assertListEqual(r2.get(), res2)
        self.assertListEqual(r3.get(), res3)
        self.assertListEqual(r4.get(), res4)
        signal = Signal2D()
        r1 = signal.flip()
        self.assertListEqual(r1.get(), [])

    def test_signal2d_to_real(self):

        s1 = Signal2D(y=[[1-1j, 1+2j],
                         [-3j,  5]])
        s2 = Signal2D(y=[[0, -3],
                         [2,  1]])
        s3 = Signal2D()
        res1 = [[1, 1], [0, 5]]
        res2 = [[0,-3], [2, 1]]
        res3 = []
        r1 = s1.to_real()
        r2 = s2.to_real()
        r3 = s3.to_real()
        self.assertIsInstance(r1, Signal)
        self.assertListEqual(r1.get(), res1)
        self.assertListEqual(r2.get(), res2)
        self.assertListEqual(r3.get(), res3)

    def test_signal2d_min(self):

        s1 = Signal2D(y=[[10, -3],
                         [4, -3.1]])
        s2 = Signal2D(y=[[3-1j, 1+2j],
                         [-3j, 5]])
        s3 = Signal2D(y=[[1]])
        s4 = Signal2D(y=[])
        s5 = Signal2D(y=[[3, 1+2j],
                         [3j, 5]])
        res1 = -3.1
        res2 = 1+2j
        res3 = 1
        r1 = s1.min()
        r2 = s2.min()
        r3 = s3.min()
        r4 = s4.min()
        self.assertEqual(r1, res1)
        self.assertEqual(r2, res2)
        self.assertEqual(r3, res3)
        self.assertIsNone(r4)
        with self.assertRaises(TypeError):
            s5.min()  # undefined

    def test_signal2d_max(self):

        s1 = Signal2D(y=[[10, -3],
                         [4, -3.1]])
        s2 = Signal2D(y=[[3-1j, 1+2j],
                         [-3j, 5]])
        s3 = Signal2D(y=[[1]])
        s4 = Signal2D(y=[])
        s5 = Signal2D(y=[[3, 1+2j],
                         [3j, 5]])
        res1 = 10
        res2 = 5
        res3 = 1
        r1 = s1.max()
        r2 = s2.max()
        r3 = s3.max()
        r4 = s4.max()
        self.assertEqual(r1, res1)
        self.assertEqual(r2, res2)
        self.assertEqual(r3, res3)
        self.assertIsNone(r4)
        with self.assertRaises(TypeError):
            s5.max()  # undefined

    def test_signal2d_energy(self):

        s1 = Signal2D(y=[[1, -2, 3],
                         [4, 5, 6],
                         [7, 8, -9]])
        s2 = Signal2D(y=[[0, 0],
                         [0, 0],
                         [0, 0]])
        s3 = Signal2D(y=[])
        s4 = Signal2D(y=[[-2]])
        self.assertEqual(s1.energy(), 285)
        self.assertEqual(s2.energy(), 0)
        self.assertIsNone(s3.energy())
        self.assertEqual(s4.power(), 4)

    def test_signal2d_power(self):

        s1 = Signal2D(y=[[1, -2, 3],
                         [4, 5, 6],
                         [7, 8, -9]])
        s2 = Signal2D(y=[[0, 0],
                         [0, 0],
                         [0, 0]])
        s3 = Signal2D(y=[])
        s4 = Signal2D(y=[[-2]])
        self.assertAlmostEqual(s1.power(), 31.6666, places=3)
        self.assertEqual(s2.power(), 0)
        self.assertIsNone(s3.power())
        self.assertEqual(s4.power(), 4)

    def test_signal2d_rms(self):

        s1 = Signal2D(y=[[1, -2, 3],
                         [4, 5, 6],
                         [7, 8, -9]])
        s2 = Signal2D(y=[[0, 0],
                         [0, 0],
                         [0, 0]])
        s3 = Signal2D(y=[])
        s4 = Signal2D(y=[[-2]])
        self.assertAlmostEqual(s1.rms(), 5.6273, places=3)
        self.assertEqual(s2.rms(), 0)
        self.assertIsNone(s3.rms())
        self.assertEqual(s4.rms(), 2)

    def test_signal2d_mean(self):

        s1 = Signal2D(y=[[1, -2, 3],
                         [4, 5, 6],
                         [7, 8, -9]])
        s2 = Signal2D(y=[[0, 0],
                         [0, 0],
                         [0, 0]])
        s3 = Signal2D(y=[])
        s4 = Signal2D(y=[[-2]])
        self.assertAlmostEqual(s1.mean(), 2.5555, places=3)
        self.assertEqual(s2.mean(), 0)
        self.assertIsNone(s3.mean())
        self.assertEqual(s4.mean(), -2)

    def test_signal2d_variance(self):

        s1 = Signal2D(y=[[1, -2, 3],
                         [4, 5, 6],
                         [7, 8, -9]])
        s2 = Signal2D(y=[[0, 0],
                         [0, 0],
                         [0, 0]])
        s3 = Signal2D(y=[])
        s4 = Signal2D(y=[[-2]])
        self.assertAlmostEqual(s1.variance(), 25.1358, places=3)
        self.assertEqual(s2.variance(), 0)
        self.assertIsNone(s3.variance())
        self.assertEqual(s4.variance(), 0)

    def test_signal2d_stddev(self):

        s1 = Signal2D(y=[[1, -2, 3],
                         [4, 5, 6],
                         [7, 8, -9]])
        s2 = Signal2D(y=[[0, 0],
                         [0, 0],
                         [0, 0]])
        s3 = Signal2D(y=[])
        s4 = Signal2D(y=[[-2]])
        self.assertAlmostEqual(s1.stddev(), 5.0135, places=3)
        self.assertEqual(s2.stddev(), 0)
        self.assertIsNone(s3.stddev())
        self.assertEqual(s4.stddev(), 0)

    def test_signal2d_mse(self):
        pass

    def test_signal2d_rmse(self):
        pass

    def test_signal2d_mae(self):
        pass

    def test_signal2d_normalize(self):

        signal = Signal2D(y=[[1, 2, 3],
                             [4, 5, 6],
                             [7, 8, 9]])
        s1 = signal.normalize()
        s2 = signal.normalize(-1, 1)
        s3 = signal.normalize(10, 100)
        s4 = signal.normalize(0.5, 1.5)
        self.assertEqual(s1.get()[0][0], 0)
        self.assertEqual(s1.get()[2][2], 1)
        self.assertEqual(s2.get()[0][0], -1)
        self.assertEqual(s2.get()[2][2], 1)
        self.assertEqual(s3.get()[0][0], 10)
        self.assertEqual(s3.get()[2][2], 100)
        self.assertEqual(s4.get()[0][0], 0.5)
        self.assertEqual(s4.get()[2][2], 1.5)
