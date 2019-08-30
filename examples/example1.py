"""
uDSP - Basic signal operations

"""

import math

from udsp.signal.builtin import (Sinewave1D, Noise1D, Gaussian2D,
                                 AudioChannel, ImageChannel)

from udsp.signal.ndim import Signal1D, Signal2D


IPATH = "examples/data/"
OPATH = "examples/data/"
AUDIOFILE = IPATH + "music.wav"
IMAGEFILE = OPATH + "cimage.png"


# Signal creation
# ===============

# Create a 3-second sine wave at 5Hz
swave = Sinewave1D(f=5, length=3, sfreq=80)

# Create normally distributed (gaussian) noise
gnoise = Noise1D(pdf="normal", length=3, sfreq=80)

# Create a custom 1D signal y(x)=x^3-4x+12 in [-3, 5]
cust = Signal1D(
    y=[x**3 - 4*x + 12 for x in range(-3, 5)],
    x=[x for x in range(-3, 5)]
)

# Create a 2D Gaussian centered at (5, 5) with std of (2, 2)
gauss = Gaussian2D(u=(5, 5), s=(2, 2),
                   length=(10, 10), sfreq=5)

# Create a list of 1D signals from the audio channels in a file
audio = AudioChannel.from_file(AUDIOFILE)

# Create a 1D signal from audio downmixed to mono
audio1 = AudioChannel.from_file(AUDIOFILE, mono=True)[0]

# Create a list of 1D signals from the color channels of an image
image = ImageChannel.from_file(IMAGEFILE)

# Create a 1D signal from a color image downmixed to grey
image1 = ImageChannel.from_file(IMAGEFILE, mono=True)[0]


# Signal arithmetic
# ==================

# Addition
nwave = swave + gnoise

# Addition with a constant
nwave2 = swave + 10

# Subtraction
swave = nwave - gnoise

# Subtraction with a constant
nwave3 = swave - 10

# Ratio
sratio = swave / gnoise

# Ratio with a constant
sratio2 = swave / 10

# Multiplication
swave2 = swave * swave

# Multiplication with a constant
swave3 = swave * 3

# Linear combination
nwave4 = swave + 0.2 * gnoise - 2 * sratio

# Scaling (by division)
nwave5 = swave + gnoise / 5

# Negation
iwave = -swave

# Exponentiation
ewave = swave ** 2


# Signal properties
# =================

# Size in samples
print(len(swave))
# or ...
print(swave.nsamples)

# Dimensions in samples
print(swave.dim)
print(image1.dim)

# Length in physical units
print(audio1.length)

# Physical units
print(audio1.xunits)

# Sampling frequency
print(audio1.sfreq)


# Signal statistics
# =================

sw_min = swave.min()
sw_max = swave.max()
sw_mean = swave.mean()
sw_stdv = swave.stddev()
sw_var = swave.variance()

sw_energy = swave.energy()
sw_power = swave.power()

# Signal transformation
# =====================

# Extract a clip from a signal
beat = audio1.clip([audio1.utos(3.4), audio1.utos(4)])

# Flip (reverse) a signal
ibeat = beat.flip()

# Extend (pad) a signal
pswave = swave.pad([swave.utos(1), swave.utos(1)])

# Normalize within an interval
naudio1 = audio1.normalize(0, 1)

# Frequency (Fourier) transform
fswave = swave.transform("frequency")

# ... or equivalently
# fswave = swave.fft()

# Power spectrum (default)
ps_swave = fswave.spectrum()

# Magnitude spectrum
ms_swave = fswave.spectrum("magnitude")

# Phase spectrum
hs_swave = fswave.spectrum("phase")

# Inverse transform
iswave = fswave.transform("timespace").to_real()

# ... or, equivalently
# iswave = fswave.ifft()

print(all(map(math.isclose, swave.get(), iswave.get())))

