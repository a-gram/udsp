# A (very) small pure Python library for basic Digital Signal Processing.

uDSP (micro-dsp) is a small library to perform basic Digital Signal Processing
operations, such as signal arithmetic, Fourier Transform, frequency response, 
filtering, etc. The library is exclusively released as an educational material 
and not as a production tool. In no way it tries to replace industry-standard 
DSP libraries such as SciPy, NumPy, etc. It has been developed as an additional 
resource to some blog articles about image processing with the sole purpose of 
illustrating DSP concepts with a fluent syntax that is much closer to plain 
English than to formal mathematical jargon.
If you find that the library uses a too much verbose syntax and could have
been made more concise, that is completely intentional.

Beware that since it is a 100% pure Python library its performance in terms 
of speed is not great.

## Requirements

The library has been developed and tested with Python 3.5. I have not tested
on earlier versions (and i'm not planning to) but probably it will work with
all 3.x versions.

The core itself does not have any dependency on third-party libraries. However,
if you intend to use the plotting functionality then it will require Matplotlib.

## Installation

Installation is trivial. Just download the package, cd into the root directory
and issue the following command

    pip install .

or the following command to install it in edit/dev mode if you want to modify
it or play around with the code

    pip install -e .

To uninstall it, use the following command

    pip uninstall udsp
    
That's it.
