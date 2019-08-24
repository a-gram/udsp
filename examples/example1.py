"""
This module shows some examples of basic operations
performed on signals.

"""

from udsp.signal import builtin as usig

# Signal creation

pms = {"sfreq": 80, "length": 3}
ipath = "examples/data/flower.png"

s = usig.Sinewave1D(f=5, **pms)
n = usig.Noise1D(**pms)

# Create a gray image from file
img = usig.GrayImage(ipath)

# Signals arithmetic

# Addition
sn = s + n

# Scaling (by multiplication)
sn2 = s + 0.2 * n
# sn2.plot.set([[s], [n], [sn2]]).graph()

# Scaling (by division)
sn2 = s + n / 5

# Subtraction
s = sn2 - 0.2 * n

# Inversion
si = -s

# Exponentiation
se = s**2

# Signal properties

# Length in samples
print(len(s))
# or ...
print(s.nsamples)
