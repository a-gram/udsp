# A (very) small pure Python library for basic Digital Signal Processing.

uDSP (micro-dsp) is a small library to perform basic Digital Signal Processing
operations, such as signal arithmetic, Fourier Transform, frequency response, 
filtering, etc. The library is exclusively released as an educational material 
or experimentation playground. It is not a production tool and in no way it 
tries to replace industry-standard DSP libraries such as SciPy, NumPy, etc. 
Being a 100% pure Python library you can expect its performance not to be great.

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
