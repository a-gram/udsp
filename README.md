# A (very) small pure Python library for basic Digital Signal Processing.

uDSP is a small (as in "tiny") library to perform basic operations that are
common in Digital Signal Processing, such as signal arithmetic, Fourier
Transform, frequency response, filtering, etc. The library is exclusively 
intended as an educational material and not as a production tool. In no way 
it tries to replace industry-standard DSP libraries such as SciPy, NumPy, etc. 
It has been developed as an additional accompanying material to some blog 
articles about image processing and its main purpose is that of illustrating 
DSP concepts with a fluent syntax that is much closer to plain English than to 
formal mathematical jargon.
If you find that the library uses a too much verbose syntax and could have
been made shorter and more concise, that is completely intentional.

Beware that since it is a 100% pure Python library its performances in terms 
of speed is not great, not surprisingly.

## Requirements

The core itself does not have any dependency on third-party libraries. However,
if you intend to use the plotting functionality (as you probably do) then it
will require Matplotlib (and its dependencies).
