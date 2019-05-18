"""
A (very) small pure Python library for basic Digital Signal Processing.

This library is exclusively intended as an educational material and not as
a professional tool. In no way it tries to replace industry-standard DSP
libraries such as SciPy, NumPy, etc. It has been developed as an additional
accompanying material to some blog articles about image processing and its
main purpose is that of illustrating DSP concepts with a fluent syntax that
is much closer to plain English than to formal mathematical jargon.
If you find that the library uses a too much verbose syntax and could have
been made shorter and more concise, that is completely intentional.

Since it is a 100% pure Python library its performances in terms of speed is
not great, as one would expect for such computational-intensive applications.


"""

from . import signal
from . import system
