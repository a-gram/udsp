"""
uDSP examples

"""

from udsp.signal.builtin import (Sinewave1D, Noise1D, Gaussian2D,
                                 AudioChannel, ImageChannel)

from udsp.core.plotter import Plotter1D, Plotter2D


IPATH = "examples/data/"
OPATH = "examples/data/"

vol = 0.6
# Create 3 notes
la = Sinewave1D(f=440, length=1, sfreq=8000)
do = Sinewave1D(f=523, length=1, sfreq=8000)
mi = Sinewave1D(f=659, length=1, sfreq=8000)
# Make the chord
lam = la + do + mi
# Make an 8-bit integer audio channel
lam8 = AudioChannel(round(lam.normalize(0, 255) * vol), bps=8)
# Save to file
AudioChannel.to_file(OPATH + "lam.wav", lam8)

