"""
This module shows some examples of basic operations
performed on signals.

"""

from udsp.signal.builtin import (Sinewave1D, Noise1D, Gaussian2D,
                                 AudioChannel, ImageChannel,
                                 MonoAudio, GrayImage)

from udsp.signal.ndim import Signal1D, Signal2D


# Signal creation
# ===============

ipath = "examples/data/"
opath = "examples/data/"

# Create a 3-second sine wave at 5Hz and some Gaussian noise
s = Sinewave1D(f=5, length=3, sfreq=80)
n = Noise1D(pdf="normal", length=3, sfreq=80)

# Create 3 notes
la = Sinewave1D(f=440, length=1, sfreq=8000)
do = Sinewave1D(f=523, length=1, sfreq=8000)
mi = Sinewave1D(f=659, length=1, sfreq=8000)

# Create a custom 1D signal y(x)=x^3-4x+12 in [-3, 5]
cust = Signal1D(
    y=[x**3 - 4*x + 12 for x in range(-3, 5)],
    x=[x for x in range(-3, 5)]
)

# Create a 2D Gaussian centered at (5, 5) with std of (2, 2)
gauss = Gaussian2D(u=(5, 5),
                   s=(2, 2),
                   length=(10, 10), sfreq=5)

# Create 1D signals from the audio channels in a file
audio = AudioChannel.from_file(ipath + "music.wav")

# Create a 1D signal from audio downmixed to mono
audio1 = MonoAudio(ipath + "music.wav")

# Create 2D signals from the color channels of an image
image = ImageChannel.from_file(ipath + "image.png")

# Create a 1D signal from color image downmixed to grey
image1 = GrayImage(ipath + "image.png")


# Signals arithmetic
# ==================

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
