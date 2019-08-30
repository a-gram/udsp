"""
uDSP - Miscellaneous examples

"""

from udsp.signal.builtin import (Sinewave1D, Noise1D, Gaussian1D,
                                 AudioChannel, ImageChannel)

from udsp.signal.ndim import Signal1D


IPATH = "examples/data/"
OPATH = "examples/data/"


# This example shows the application of signal arithmetic
# for denoising using the "ensemble averaging" method.

NITER = 20

# Signal's parameters
sparams = dict(length=5, sfreq=50)

# Create a signal with two peaks at 1.5 and 3.5
signal = (Gaussian1D(u=1.5, s=0.3, **sparams) +
          Gaussian1D(u=3.5, s=0.2, k=0.6, **sparams))

# Accumulator
nsignal = Signal1D([0] * len(signal), **sparams)

snr = []
for i in range(NITER):
    gnoise = Noise1D(pdf="normal", **sparams)
    nsignal = (nsignal + (signal + 0.5 * gnoise) / NITER)
    snr.append(nsignal.mean() / nsignal.stddev())

# Verify that the SNR increases
Signal1D(snr).plot.graph()


# Working with multimedia data (audio, image)
# ===========================================

# Audio and image data are manipulated by means of the
# AudioChannel and ImageChannel classes in the 'builtin'
# package. As the name implies, an instance of such classes
# represents a single audio/image data channel and provides
# all the operations supported by a signal object.
#
# To load audio/image data from a file, use the class methods
# provided by the Audio/ImageChannel interface, as follows
#
# audio = AudioChannel.from_file(AUDIOFILE)
# image = ImageChannel.from_file(IMAGEFILE)
#
# The from_file() method returns a list of audio/image channel
# objects. To save audio/image data to a file, use the
# to_file() class method, as follows
#
# AudioChannel.to_file(AUDIOFILE, audio)
# ImageChannel.to_file(IMAGEFILE, image)
#
# where the audio/image argument is a list of Audio/ImageChannel
# objects.
#
# Currently, only PNG images and WAV audio are supported.
#

# This example creates an audio signal by combining 3
# tones at different frequencies so that they can form
# a chord (Am), and then save it to file.

vol = 0.6
# Signal's parameters
sparams = dict(length=2, sfreq=8000)
# Note frequencies
nf = [440, 523, 659]
# Create 3 notes with 2-second duration sampled at 8KHz
notes = map(lambda f: Sinewave1D(f=f, **sparams), nf)
# Make the chord
lam = sum(notes)
# Create an 8-bit integer audio channel
lam8 = AudioChannel(round(lam.normalize(0, 255) * vol), bps=8)
# Save to file
AudioChannel.to_file(OPATH + "lam.wav", lam8)


# This example loads a colour (RGB) image and increases
# its luminosity. The resulting image is saved to a file.

lum = 0.5
# Load the image's colour channels
image = ImageChannel.from_file(IPATH + "cimage.png")
# bimage = list(map(operator.mul, image, [lum] * len(image)))
# Scale the intensity values
bimage = [round(i * lum) for i in image]
# Save to a PNG file
ImageChannel.to_file(OPATH + "image_out.png", bimage)

