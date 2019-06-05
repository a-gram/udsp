"""
PNG image implementation

"""

import collections
import io
import itertools
import math
import operator
import re
import struct
import warnings
import zlib

from array import array
from .base import ImageCodec, Metadata


class PNGCodec(ImageCodec):

    format = "png"
    description = "Portable Network Graphics coder/encoder"

    def __init__(self, stream):
        super().__init__(stream)
        self._reader = PNGReader(stream)
        self._writer = None

    def decode(self):
        data = self._reader.read()
        return data[2]

    def encode(self):
        raise NotImplementedError

    def get_metadata(self):
        if not hasattr(self._reader, "width"):
            self._reader.preamble()
        meta = Metadata()
        meta.size = (self._reader.width, self._reader.height)
        meta.bps = self._reader.bitdepth
        meta.planes = self._reader.planes
        return meta


# PNG encoder/decoder in pure Python
#
# Copyright (C) 2006 Johann C. Rocholl <johann@browsershots.org>
# Portions Copyright (C) 2009 David Jones <drj@pobox.com>
# And probably portions Copyright (C) 2006 Nicko van Someren <nicko@nicko.org>
#
# Original concept by Johann C. Rocholl.
#
# LICENCE (MIT)
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

__version__ = "0.0.19"

signature = struct.pack('8B', 137, 80, 78, 71, 13, 10, 26, 10)

# The xstart, ystart, xstep, ystep for the Adam7 interlace passes.
adam7 = ((0, 0, 8, 8),
         (4, 0, 8, 8),
         (0, 4, 4, 8),
         (2, 0, 4, 4),
         (0, 2, 2, 4),
         (1, 0, 2, 2),
         (0, 1, 1, 2))


def adam7_generate(width, height):
    """
    Generate the coordinates for the reduced scanlines
    of an Adam7 interlaced image
    of size `width` by `height` pixels.

    Yields a generator for each pass,
    and each pass generator yields a series of (x, y, xstep) triples,
    each one identifying a reduced scanline consisting of
    pixels starting at (x, y) and taking every xstep pixel to the right.
    """

    for xstart, ystart, xstep, ystep in adam7:
        if xstart >= width:
            continue
        yield ((xstart, y, xstep) for y in range(ystart, height, ystep))


# Models the 'pHYs' chunk (used by the Reader)
Resolution = collections.namedtuple('_Resolution', 'x y unit_is_meter')


def group(s, n):
    return list(zip(* [iter(s)] * n))


def isarray(x):
    return isinstance(x, array)


def check_palette(palette):
    """
    Check a palette argument (to the :class:`Writer` class) for validity.
    Returns the palette as a list if okay;
    raises an exception otherwise.
    """

    # None is the default and is allowed.
    if palette is None:
        return None

    p = list(palette)
    if not (0 < len(p) <= 256):
        raise ProtocolError(
            "a palette must have between 1 and 256 entries,"
            " see https://www.w3.org/TR/PNG/#11PLTE")
    seen_triple = False
    for i, t in enumerate(p):
        if len(t) not in (3, 4):
            raise ProtocolError(
                "palette entry %d: entries must be 3- or 4-tuples." % i)
        if len(t) == 3:
            seen_triple = True
        if seen_triple and len(t) == 4:
            raise ProtocolError(
                "palette entry %d: all 4-tuples must precede all 3-tuples" % i)
        for x in t:
            if int(x) != x or not(0 <= x <= 255):
                raise ProtocolError(
                    "palette entry %d: "
                    "values must be integer: 0 <= x <= 255" % i)
    return p


def check_sizes(size, width, height):
    """
    Check that these arguments, if supplied, are consistent.
    Return a (width, height) pair.
    """

    if not size:
        return width, height

    if len(size) != 2:
        raise ProtocolError(
            "size argument should be a pair (width, height)")
    if width is not None and width != size[0]:
        raise ProtocolError(
            "size[0] (%r) and width (%r) should match when both are used."
            % (size[0], width))
    if height is not None and height != size[1]:
        raise ProtocolError(
            "size[1] (%r) and height (%r) should match when both are used."
            % (size[1], height))
    return size


def check_color(c, greyscale, which):
    """
    Checks that a colour argument for transparent or background options
    is the right form.
    Returns the colour
    (which, if it's a bare integer, is "corrected" to a 1-tuple).
    """

    if c is None:
        return c
    if greyscale:
        try:
            len(c)
        except TypeError:
            c = (c,)
        if len(c) != 1:
            raise ProtocolError("%s for greyscale must be 1-tuple" % which)
        if not is_natural(c[0]):
            raise ProtocolError(
                "%s colour for greyscale must be integer" % which)
    else:
        if not (len(c) == 3 and
                is_natural(c[0]) and
                is_natural(c[1]) and
                is_natural(c[2])):
            raise ProtocolError(
                "%s colour must be a triple of integers" % which)
    return c


class Error(Exception):
    def __str__(self):
        return self.__class__.__name__ + ': ' + ' '.join(self.args)


class FormatError(Error):
    """
    Problem with input file format.
    In other words, PNG file does not conform to
    the specification in some way and is invalid.
    """


class ProtocolError(Error):
    """
    Problem with the way the programming interface has been used,
    or the data presented to it.
    """


class ChunkError(FormatError):
    pass


class Default:
    """The default for the greyscale paramter."""


def write_chunk(outfile, tag, data=b''):
    """
    Write a PNG chunk to the output file, including length and
    checksum.
    """

    data = bytes(data)
    # http://www.w3.org/TR/PNG/#5Chunk-layout
    outfile.write(struct.pack("!I", len(data)))
    outfile.write(tag)
    outfile.write(data)
    checksum = zlib.crc32(tag)
    checksum = zlib.crc32(data, checksum)
    checksum &= 2 ** 32 - 1
    outfile.write(struct.pack("!I", checksum))


def write_chunks(out, chunks):
    """Create a PNG file by writing out the chunks."""

    out.write(signature)
    for chunk in chunks:
        write_chunk(out, *chunk)


def rescale_rows(rows, rescale):
    """
    Take each row in rows (an iterator) and yield
    a fresh row with the pixels scaled according to
    the rescale parameters in the list `rescale`.
    Each element of `rescale` is a tuple of
    (source_bitdepth, target_bitdepth),
    with one element per channel.
    """

    # One factor for each channel
    fs = [float(2 ** s[1] - 1)/float(2 ** s[0] - 1)
          for s in rescale]

    # Assume all target_bitdepths are the same
    target_bitdepths = set(s[1] for s in rescale)
    assert len(target_bitdepths) == 1
    (target_bitdepth, ) = target_bitdepths
    typecode = 'BH'[target_bitdepth > 8]

    # Number of channels
    n_chans = len(rescale)

    for row in rows:
        rescaled_row = array(typecode, iter(row))
        for i in range(n_chans):
            channel = array(
                typecode,
                (int(round(fs[i] * x)) for x in row[i::n_chans]))
            rescaled_row[i::n_chans] = channel
        yield rescaled_row


def pack_rows(rows, bitdepth):
    """Yield packed rows that are a byte array.
    Each byte is packed with the values from several pixels.
    """

    assert bitdepth < 8
    assert 8 % bitdepth == 0

    # samples per byte
    spb = int(8 / bitdepth)

    def make_byte(block):
        """Take a block of (2, 4, or 8) values,
        and pack them into a single byte.
        """

        res = 0
        for v in block:
            res = (res << bitdepth) + v
        return res

    for row in rows:
        a = bytearray(row)
        # Adding padding bytes so we can group into a whole
        # number of spb-tuples.
        n = float(len(a))
        extra = math.ceil(n / spb) * spb - n
        a.extend([0] * int(extra))
        # Pack into bytes.
        # Each block is the samples for one byte.
        blocks = group(a, spb)
        yield bytearray(make_byte(block) for block in blocks)


def unpack_rows(rows):
    """Unpack each row from being 16-bits per value,
    to being a sequence of bytes.
    """
    for row in rows:
        fmt = '!%dH' % len(row)
        yield bytearray(struct.pack(fmt, *row))


def make_palette_chunks(palette):
    """
    Create the byte sequences for a ``PLTE`` and
    if necessary a ``tRNS`` chunk.
    Returned as a pair (*p*, *t*).
    *t* will be ``None`` if no ``tRNS`` chunk is necessary.
    """

    p = bytearray()
    t = bytearray()

    for x in palette:
        p.extend(x[0:3])
        if len(x) > 3:
            t.append(x[3])
    if t:
        return p, t
    return p, None


def check_bitdepth_rescale(
        palette, bitdepth, transparent, alpha, greyscale):
    """
    Returns (bitdepth, rescale) pair.
    """

    if palette:
        if len(bitdepth) != 1:
            raise ProtocolError(
                "with palette, only a single bitdepth may be used")
        (bitdepth, ) = bitdepth
        if bitdepth not in (1, 2, 4, 8):
            raise ProtocolError(
                "with palette, bitdepth must be 1, 2, 4, or 8")
        if transparent is not None:
            raise ProtocolError("transparent and palette not compatible")
        if alpha:
            raise ProtocolError("alpha and palette not compatible")
        if greyscale:
            raise ProtocolError("greyscale and palette not compatible")
        return bitdepth, None

    # No palette, check for sBIT chunk generation.

    if greyscale and not alpha:
        # Single channel, L.
        (bitdepth,) = bitdepth
        if bitdepth in (1, 2, 4, 8, 16):
            return bitdepth, None
        if bitdepth > 8:
            targetbitdepth = 16
        elif bitdepth == 3:
            targetbitdepth = 4
        else:
            assert bitdepth in (5, 6, 7)
            targetbitdepth = 8
        return targetbitdepth, [(bitdepth, targetbitdepth)]

    assert alpha or not greyscale

    depth_set = tuple(set(bitdepth))
    if depth_set in [(8,), (16,)]:
        # No sBIT required.
        (bitdepth, ) = depth_set
        return bitdepth, None

    targetbitdepth = (8, 16)[max(bitdepth) > 8]
    return targetbitdepth, [(b, targetbitdepth) for b in bitdepth]


# Regex for decoding mode string
RegexModeDecode = re.compile("(LA?|RGBA?);?([0-9]*)", flags=re.IGNORECASE)


class PNGReader:
    """
    Pure Python PNG decoder in pure Python.
    """

    def __init__(self, _guess=None, filename=None, file=None, bytez=None):
        """
        The constructor expects exactly one keyword argument.
        If you supply a positional argument instead,
        it will guess the input type.
        Choose from the following keyword arguments:

        filename
          Name of input file (a PNG file).
        file
          A file-like object (object with a read() method).
        bytes
          ``bytes`` or ``bytearray`` with PNG data.

        """
        keywords_supplied = (
            (_guess is not None) +
            (filename is not None) +
            (file is not None) +
            (bytez is not None))
        if keywords_supplied != 1:
            raise TypeError("Reader() takes exactly 1 argument")

        # Will be the first 8 bytes, later on.  See validate_signature.
        self.signature = None
        self.transparent = None
        # A pair of (len,type) if a chunk has been read but its data and
        # checksum have not (in other words the file position is just
        # past the 4 bytes that specify the chunk type).
        # See preamble method for how this is used.
        self.atchunk = None

        if _guess is not None:
            if isarray(_guess):
                bytez = _guess
            elif isinstance(_guess, str):
                filename = _guess
            elif hasattr(_guess, 'read'):
                file = _guess

        if bytez is not None:
            self.file = io.BytesIO(bytez)
        elif filename is not None:
            self.file = open(filename, "rb")
        elif file is not None:
            self.file = file
        else:
            raise ProtocolError("expecting filename, file or bytes array")

    def chunk(self, lenient=False):
        """
        Read the next PNG chunk from the input file;
        returns a (*type*, *data*) tuple.
        *type* is the chunk's type as a byte string
        (all PNG chunk types are 4 bytes long).
        *data* is the chunk's data content, as a byte string.

        If the optional `lenient` argument evaluates to `True`,
        checksum failures will raise warnings rather than exceptions.
        """

        self.validate_signature()

        # http://www.w3.org/TR/PNG/#5Chunk-layout
        if not self.atchunk:
            self.atchunk = self._chunk_len_type()
        if not self.atchunk:
            raise ChunkError("No more chunks.")
        length, ctype = self.atchunk
        self.atchunk = None

        data = self.file.read(length)
        if len(data) != length:
            raise ChunkError(
                'Chunk %s too short for required %i octets.'
                % (ctype, length))
        checksum = self.file.read(4)
        if len(checksum) != 4:
            raise ChunkError('Chunk %s too short for checksum.' % ctype)
        verify = zlib.crc32(ctype)
        verify = zlib.crc32(data, verify)
        # Whether the output from zlib.crc32 is signed or not varies
        # according to hideous implementation details, see
        # http://bugs.python.org/issue1202 .
        # We coerce it to be positive here (in a way which works on
        # Python 2.3 and older).
        verify &= 2**32 - 1
        verify = struct.pack('!I', verify)
        if checksum != verify:
            (a, ) = struct.unpack('!I', checksum)
            (b, ) = struct.unpack('!I', verify)
            message = ("Checksum error in %s chunk: 0x%08X != 0x%08X."
                       % (ctype.decode('ascii'), a, b))
            if lenient:
                warnings.warn(message, RuntimeWarning)
            else:
                raise ChunkError(message)
        return ctype, data

    def chunks(self):
        """Return an iterator that will yield each chunk as a
        (*chunktype*, *content*) pair.
        """

        while True:
            t, v = self.chunk()
            yield t, v
            if t == b'IEND':
                break

    def undo_filter(self, filter_type, scanline, previous):
        """
        Undo the filter for a scanline.
        `scanline` is a sequence of bytes that
        does not include the initial filter type byte.
        `previous` is decoded previous scanline
        (for straightlaced images this is the previous pixel row,
        but for interlaced images, it is
        the previous scanline in the reduced image,
        which in general is not the previous pixel row in the final image).
        When there is no previous scanline
        (the first row of a straightlaced image,
        or the first row in one of the passes in an interlaced image),
        then this argument should be ``None``.

        The scanline will have the effects of filtering removed;
        the result will be returned as a fresh sequence of bytes.
        """

        # :todo: Would it be better to update scanline in place?
        result = scanline

        if filter_type == 0:
            return result

        if filter_type not in (1, 2, 3, 4):
            raise FormatError(
                'Invalid PNG Filter Type.  '
                'See http://www.w3.org/TR/2003/REC-PNG-20031110/#9Filters .')

        # Filter unit.  The stride from one pixel to the corresponding
        # byte from the previous pixel.  Normally this is the pixel
        # size in bytes, but when this is smaller than 1, the previous
        # byte is used instead.
        fu = max(1, self.psize)

        # For the first line of a pass, synthesize a dummy previous
        # line.  An alternative approach would be to observe that on the
        # first line 'up' is the same as 'null', 'paeth' is the same
        # as 'sub', with only 'average' requiring any special case.
        if not previous:
            previous = bytearray([0] * len(scanline))

        # Call appropriate filter algorithm.  Note that 0 has already
        # been dealt with.
        fn = (None,
              undo_filter_sub,
              undo_filter_up,
              undo_filter_average,
              undo_filter_paeth)[filter_type]
        fn(fu, scanline, previous, result)
        return result

    def _deinterlace(self, raw):
        """
        Read raw pixel data, undo filters, deinterlace, and flatten.
        Return a single array of values.
        """

        # Values per row (of the target image)
        vpr = self.width * self.planes

        # Values per image
        vpi = vpr * self.height
        # Interleaving writes to the output array randomly
        # (well, not quite), so the entire output array must be in memory.
        # Make a result array, and make it big enough.
        if self.bitdepth > 8:
            a = array('H', [0] * vpi)
        else:
            a = bytearray([0] * vpi)
        source_offset = 0

        for lines in adam7_generate(self.width, self.height):
            # The previous (reconstructed) scanline.
            # `None` at the beginning of a pass
            # to indicate that there is no previous line.
            recon = None
            for x, y, xstep in lines:
                # Pixels per row (reduced pass image)
                ppr = int(math.ceil((self.width - x) / float(xstep)))
                # Row size in bytes for this pass.
                row_size = int(math.ceil(self.psize * ppr))

                filter_type = raw[source_offset]
                source_offset += 1
                scanline = raw[source_offset: source_offset + row_size]
                source_offset += row_size
                recon = self.undo_filter(filter_type, scanline, recon)
                # Convert so that there is one element per pixel value
                flat = self._bytes_to_values(recon, width=ppr)
                if xstep == 1:
                    assert x == 0
                    offset = y * vpr
                    a[offset: offset + vpr] = flat
                else:
                    offset = y * vpr + x * self.planes
                    end_offset = (y + 1) * vpr
                    skip = self.planes * xstep
                    for i in range(self.planes):
                        a[offset + i: end_offset: skip] = \
                            flat[i:: self.planes]

        return a

    def _iter_bytes_to_values(self, byte_rows):
        """
        Iterator that yields each scanline;
        each scanline being a sequence of values.
        `byte_rows` should be an iterator that yields
        the bytes of each row in turn.
        """

        for row in byte_rows:
            yield self._bytes_to_values(row)

    def _bytes_to_values(self, bs, width=None):
        """Convert a packed row of bytes into a row of values.
        Result will be a freshly allocated object,
        not shared with the argument.
        """

        if self.bitdepth == 8:
            return bytearray(bs)
        if self.bitdepth == 16:
            return array('H',
                         struct.unpack('!%dH' % (len(bs) // 2), bs))

        assert self.bitdepth < 8
        if width is None:
            width = self.width
        # Samples per byte
        spb = 8 // self.bitdepth
        out = bytearray()
        mask = 2**self.bitdepth - 1
        shifts = [self.bitdepth * i
                  for i in reversed(list(range(spb)))]
        for o in bs:
            out.extend([mask & (o >> i) for i in shifts])
        return out[:width]

    def _iter_straight_packed(self, byte_blocks):
        """Iterator that undoes the effect of filtering;
        yields each row as a sequence of packed bytes.
        Assumes input is straightlaced.
        `byte_blocks` should be an iterable that yields the raw bytes
        in blocks of arbitrary size.
        """

        # length of row, in bytes
        rb = self.row_bytes
        a = bytearray()
        # The previous (reconstructed) scanline.
        # None indicates first line of image.
        recon = None
        for some_bytes in byte_blocks:
            a.extend(some_bytes)
            while len(a) >= rb + 1:
                filter_type = a[0]
                scanline = a[1: rb + 1]
                del a[: rb + 1]
                recon = self.undo_filter(filter_type, scanline, recon)
                yield recon
        if len(a) != 0:
            # :file:format We get here with a file format error:
            # when the available bytes (after decompressing) do not
            # pack into exact rows.
            raise FormatError('Wrong size for decompressed IDAT chunk.')
        assert len(a) == 0

    def validate_signature(self):
        """
        If signature (header) has not been read then read and
        validate it; otherwise do nothing.
        """

        if self.signature:
            return
        self.signature = self.file.read(8)
        if self.signature != signature:
            raise FormatError("PNG file has invalid signature.")

    def preamble(self, lenient=False):
        """
        Extract the image metadata by reading
        the initial part of the PNG file up to
        the start of the ``IDAT`` chunk.
        All the chunks that precede the ``IDAT`` chunk are
        read and either processed for metadata or discarded.

        If the optional `lenient` argument evaluates to `True`,
        checksum failures will raise warnings rather than exceptions.
        """

        self.validate_signature()

        while True:
            if not self.atchunk:
                self.atchunk = self._chunk_len_type()
                if self.atchunk is None:
                    raise FormatError('This PNG file has no IDAT chunks.')
            if self.atchunk[1] == b'IDAT':
                return
            self.process_chunk(lenient=lenient)

    def _chunk_len_type(self):
        """
        Reads just enough of the input to
        determine the next chunk's length and type;
        return a (*length*, *type*) pair where *type* is a byte sequence.
        If there are no more chunks, ``None`` is returned.
        """

        x = self.file.read(8)
        if not x:
            return None
        if len(x) != 8:
            raise FormatError(
                'End of file whilst reading chunk length and type.')
        length, ctype = struct.unpack('!I4s', x)
        if length > 2 ** 31 - 1:
            raise FormatError('Chunk %s is too large: %d.' % (ctype, length))
        # Check that all bytes are in valid ASCII range.
        # https://www.w3.org/TR/2003/REC-PNG-20031110/#5Chunk-layout
        type_bytes = set(bytearray(ctype))
        if not(type_bytes <= set(range(65, 91)) | set(range(97, 123))):
            raise FormatError(
                'Chunk %r has invalid Chunk Type.'
                % list(ctype))
        return length, ctype

    def process_chunk(self, lenient=False):
        """
        Process the next chunk and its data.
        This only processes the following chunk types:
        ``IHDR``, ``PLTE``, ``bKGD``, ``tRNS``, ``gAMA``, ``sBIT``, ``pHYs``.
        All other chunk types are ignored.

        If the optional `lenient` argument evaluates to `True`,
        checksum failures will raise warnings rather than exceptions.
        """

        ctype, data = self.chunk(lenient=lenient)
        method = '_process_' + ctype.decode('ascii')
        m = getattr(self, method, None)
        if m:
            m(data)

    def _process_IHDR(self, data):
        # http://www.w3.org/TR/PNG/#11IHDR
        if len(data) != 13:
            raise FormatError('IHDR chunk has incorrect length.')
        (self.width, self.height, self.bitdepth, self.color_type,
         self.compression, self.filter,
         self.interlace) = struct.unpack("!2I5B", data)

        check_bitdepth_colortype(self.bitdepth, self.color_type)

        if self.compression != 0:
            raise FormatError(
                "Unknown compression method %d" % self.compression)
        if self.filter != 0:
            raise FormatError(
                "Unknown filter method %d,"
                " see http://www.w3.org/TR/2003/REC-PNG-20031110/#9Filters ."
                % self.filter)
        if self.interlace not in (0, 1):
            raise FormatError(
                "Unknown interlace method %d, see "
                "http://www.w3.org/TR/2003/REC-PNG-20031110/#8InterlaceMethods"
                " ."
                % self.interlace)

        # Derived values
        # http://www.w3.org/TR/PNG/#6Colour-values
        colormap = bool(self.color_type & 1)
        greyscale = not(self.color_type & 2)
        alpha = bool(self.color_type & 4)
        color_planes = (3, 1)[greyscale or colormap]
        planes = color_planes + alpha

        self.colormap = colormap
        self.greyscale = greyscale
        self.alpha = alpha
        self.color_planes = color_planes
        self.planes = planes
        self.psize = float(self.bitdepth) / float(8) * planes
        if int(self.psize) == self.psize:
            self.psize = int(self.psize)
        self.row_bytes = int(math.ceil(self.width * self.psize))
        # Stores PLTE chunk if present, and is used to check
        # chunk ordering constraints.
        self.plte = None
        # Stores tRNS chunk if present, and is used to check chunk
        # ordering constraints.
        self.trns = None
        # Stores sBIT chunk if present.
        self.sbit = None

    def _process_PLTE(self, data):
        # http://www.w3.org/TR/PNG/#11PLTE
        if self.plte:
            warnings.warn("Multiple PLTE chunks present.")
        self.plte = data
        if len(data) % 3 != 0:
            raise FormatError(
                "PLTE chunk's length should be a multiple of 3.")
        if len(data) > (2 ** self.bitdepth) * 3:
            raise FormatError("PLTE chunk is too long.")
        if len(data) == 0:
            raise FormatError("Empty PLTE is not allowed.")

    def _process_bKGD(self, data):
        try:
            if self.colormap:
                if not self.plte:
                    warnings.warn(
                        "PLTE chunk is required before bKGD chunk.")
                self.background = struct.unpack('B', data)
            else:
                self.background = struct.unpack("!%dH" % self.color_planes,
                                                data)
        except struct.error:
            raise FormatError("bKGD chunk has incorrect length.")

    def _process_tRNS(self, data):
        # http://www.w3.org/TR/PNG/#11tRNS
        self.trns = data
        if self.colormap:
            if not self.plte:
                warnings.warn("PLTE chunk is required before tRNS chunk.")
            else:
                if len(data) > len(self.plte) / 3:
                    # Was warning, but promoted to Error as it
                    # would otherwise cause pain later on.
                    raise FormatError("tRNS chunk is too long.")
        else:
            if self.alpha:
                raise FormatError(
                    "tRNS chunk is not valid with colour type %d." %
                    self.color_type)
            try:
                self.transparent = \
                    struct.unpack("!%dH" % self.color_planes, data)
            except struct.error:
                raise FormatError("tRNS chunk has incorrect length.")

    def _process_gAMA(self, data):
        try:
            self.gamma = struct.unpack("!L", data)[0] / 100000.0
        except struct.error:
            raise FormatError("gAMA chunk has incorrect length.")

    def _process_sBIT(self, data):
        self.sbit = data
        if (self.colormap and len(data) != 3 or
                not self.colormap and len(data) != self.planes):
            raise FormatError("sBIT chunk has incorrect length.")

    def _process_pHYs(self, data):
        # http://www.w3.org/TR/PNG/#11pHYs
        self.phys = data
        fmt = "!LLB"
        if len(data) != struct.calcsize(fmt):
            raise FormatError("pHYs chunk has incorrect length.")
        self.x_pixels_per_unit, self.y_pixels_per_unit, unit = \
            struct.unpack(fmt, data)
        self.unit_is_meter = bool(unit)

    def read(self, lenient=False):
        """
        Read the PNG file and decode it.
        Returns (`width`, `height`, `rows`, `info`).

        May use excessive memory.

        `rows` is a sequence of rows;
        each row is a sequence of values.

        If the optional `lenient` argument evaluates to True,
        checksum failures will raise warnings rather than exceptions.
        """

        def iteridat():
            """Iterator that yields all the ``IDAT`` chunks as strings."""
            while True:
                ctype, data = self.chunk(lenient=lenient)
                if ctype == b'IEND':
                    # http://www.w3.org/TR/PNG/#11IEND
                    break
                if ctype != b'IDAT':
                    continue
                # type == b'IDAT'
                # http://www.w3.org/TR/PNG/#11IDAT
                if self.colormap and not self.plte:
                    warnings.warn("PLTE chunk is required before IDAT chunk")
                yield data

        self.preamble(lenient=lenient)
        raw = decompress(iteridat())

        if self.interlace:
            def rows_from_interlace():
                """Yield each row from an interlaced PNG."""
                # It's important that this iterator doesn't read
                # IDAT chunks until it yields the first row.
                bs = bytearray(itertools.chain(*raw))
                arraycode = 'BH'[self.bitdepth > 8]
                # Like :meth:`group` but
                # producing an array.array object for each row.
                values = self._deinterlace(bs)
                vpr = self.width * self.planes
                for i in range(0, len(values), vpr):
                    row = array(arraycode, values[i:i+vpr])
                    yield row
            rows = rows_from_interlace()
        else:
            rows = self._iter_bytes_to_values(self._iter_straight_packed(raw))
        info = dict()
        for attr in 'greyscale alpha planes bitdepth interlace'.split():
            info[attr] = getattr(self, attr)
        info['size'] = (self.width, self.height)
        for attr in 'gamma transparent background'.split():
            a = getattr(self, attr, None)
            if a is not None:
                info[attr] = a
        if getattr(self, 'x_pixels_per_unit', None):
            info['physical'] = Resolution(self.x_pixels_per_unit,
                                          self.y_pixels_per_unit,
                                          self.unit_is_meter)
        if self.plte:
            info['palette'] = self.palette()
        return self.width, self.height, rows, info

    def read_flat(self):
        """
        Read a PNG file and decode it into a single array of values.
        Returns (*width*, *height*, *values*, *info*).

        May use excessive memory.

        `values` is a single array.

        The :meth:`read` method is more stream-friendly than this,
        because it returns a sequence of rows.
        """

        x, y, pixel, info = self.read()
        arraycode = 'BH'[info['bitdepth'] > 8]
        pixel = array(arraycode, itertools.chain(*pixel))
        return x, y, pixel, info

    def palette(self, alpha='natural'):
        """
        Returns a palette that is a sequence of 3-tuples or 4-tuples,
        synthesizing it from the ``PLTE`` and ``tRNS`` chunks.
        These chunks should have already been processed (for example,
        by calling the :meth:`preamble` method).
        All the tuples are the same size:
        3-tuples if there is no ``tRNS`` chunk,
        4-tuples when there is a ``tRNS`` chunk.

        Assumes that the image is colour type
        3 and therefore a ``PLTE`` chunk is required.

        If the `alpha` argument is ``'force'`` then an alpha channel is
        always added, forcing the result to be a sequence of 4-tuples.
        """

        if not self.plte:
            raise FormatError(
                "Required PLTE chunk is missing in colour type 3 image.")
        plte = group(array('B', self.plte), 3)
        if self.trns or alpha == 'force':
            trns = array('B', self.trns or [])
            trns.extend([255] * (len(plte) - len(trns)))
            plte = list(map(operator.add, plte, group(trns, 1)))
        return plte

    def asDirect(self):
        """
        Returns the image data as a direct representation of
        an ``x * y * planes`` array.
        This removes the need for callers to deal with
        palettes and transparency themselves.
        Images with a palette (colour type 3) are converted to RGB or RGBA;
        images with transparency (a ``tRNS`` chunk) are converted to
        LA or RGBA as appropriate.
        When returned in this format the pixel values represent
        the colour value directly without needing to refer
        to palettes or transparency information.

        Like the :meth:`read` method this method returns a 4-tuple:

        (*width*, *height*, *rows*, *info*)

        This method normally returns pixel values with
        the bit depth they have in the source image, but
        when the source PNG has an ``sBIT`` chunk it is inspected and
        can reduce the bit depth of the result pixels;
        pixel values will be reduced according to the bit depth
        specified in the ``sBIT`` chunk.
        PNG nerds should note a single result bit depth is
        used for all channels:
        the maximum of the ones specified in the ``sBIT`` chunk.
        An RGB565 image will be rescaled to 6-bit RGB666.

        The *info* dictionary that is returned reflects
        the `direct` format and not the original source image.
        For example, an RGB source image with a ``tRNS`` chunk
        to represent a transparent colour,
        will start with ``planes=3`` and ``alpha=False`` for the
        source image,
        but the *info* dictionary returned by this method
        will have ``planes=4`` and ``alpha=True`` because
        an alpha channel is synthesized and added.

        *rows* is a sequence of rows;
        each row being a sequence of values
        (like the :meth:`read` method).

        All the other aspects of the image data are not changed.
        """

        self.preamble()

        # Simple case, no conversion necessary.
        if not self.colormap and not self.trns and not self.sbit:
            return self.read()

        x, y, pixels, info = self.read()

        if self.colormap:
            info['colormap'] = False
            info['alpha'] = bool(self.trns)
            info['bitdepth'] = 8
            info['planes'] = 3 + bool(self.trns)
            plte = self.palette()

            def iterpal(ipixels):
                for row in ipixels:
                    row = [plte[r] for r in row]
                    yield array('B', itertools.chain(*row))
            pixels = iterpal(pixels)
        elif self.trns:
            # It would be nice if there was some reasonable way
            # of doing this without generating a whole load of
            # intermediate tuples.  But tuples does seem like the
            # easiest way, with no other way clearly much simpler or
            # much faster.  (Actually, the L to LA conversion could
            # perhaps go faster (all those 1-tuples!), but I still
            # wonder whether the code proliferation is worth it)
            it = self.transparent
            maxval = 2 ** info['bitdepth'] - 1
            planes = info['planes']
            info['alpha'] = True
            info['planes'] += 1
            typecode = 'BH'[info['bitdepth'] > 8]

            def itertrns(ipixels):
                for row in ipixels:
                    # For each row we group it into pixels, then form a
                    # characterisation vector that says whether each
                    # pixel is opaque or not.  Then we convert
                    # True/False to 0/maxval (by multiplication),
                    # and add it as the extra channel.
                    row = group(row, planes)
                    opa = map(it.__ne__, row)
                    opa = map(maxval.__mul__, opa)
                    opa = list(zip(opa))    # convert to 1-tuples
                    yield array(
                        typecode,
                        itertools.chain(*map(operator.add, row, opa)))
            pixels = itertrns(pixels)
        targetbitdepth = None
        if self.sbit:
            sbit = struct.unpack('%dB' % len(self.sbit), self.sbit)
            targetbitdepth = max(sbit)
            if targetbitdepth > info['bitdepth']:
                raise Error('sBIT chunk %r exceeds bitdepth %d' %
                            (sbit, self.bitdepth))
            if min(sbit) <= 0:
                raise Error('sBIT chunk %r has a 0-entry' % sbit)
        if targetbitdepth:
            shift = info['bitdepth'] - targetbitdepth
            info['bitdepth'] = targetbitdepth

            def itershift(ipixels):
                for row in ipixels:
                    yield [p >> shift for p in row]
            pixels = itershift(pixels)
        return x, y, pixels, info

    def _as_rescale(self, get, targetbitdepth):
        """Helper used by :meth:`asRGB8` and :meth:`asRGBA8`."""

        width, height, pixels, info = get()
        maxval = 2**info['bitdepth'] - 1
        targetmaxval = 2**targetbitdepth - 1
        factor = float(targetmaxval) / float(maxval)
        info['bitdepth'] = targetbitdepth

        def iterscale():
            for row in pixels:
                yield [int(round(x * factor)) for x in row]
        if maxval == targetmaxval:
            return width, height, pixels, info
        else:
            return width, height, iterscale(), info

    def asRGB8(self):
        """
        Return the image data as an RGB pixels with 8-bits per sample.
        This is like the :meth:`asRGB` method except that
        this method additionally rescales the values so that
        they are all between 0 and 255 (8-bit).
        In the case where the source image has a bit depth < 8
        the transformation preserves all the information;
        where the source image has bit depth > 8, then
        rescaling to 8-bit values loses precision.
        No dithering is performed.
        Like :meth:`asRGB`,
        an alpha channel in the source image will raise an exception.

        This function returns a 4-tuple:
        (*width*, *height*, *rows*, *info*).
        *width*, *height*, *info* are as per the :meth:`read` method.

        *rows* is the pixel data as a sequence of rows.
        """

        return self._as_rescale(self.asRGB, 8)

    def asRGBA8(self):
        """
        Return the image data as RGBA pixels with 8-bits per sample.
        This method is similar to :meth:`asRGB8` and :meth:`asRGBA`:
        The result pixels have an alpha channel, *and*
        values are rescaled to the range 0 to 255.
        The alpha channel is synthesized if necessary
        (with a small speed penalty).
        """

        return self._as_rescale(self.asRGBA, 8)

    def asRGB(self):
        """
        Return image as RGB pixels.
        RGB colour images are passed through unchanged;
        greyscales are expanded into RGB triplets
        (there is a small speed overhead for doing this).

        An alpha channel in the source image will raise an exception.

        The return values are as for the :meth:`read` method except that
        the *info* reflect the returned pixels, not the source image.
        In particular,
        for this method ``info['greyscale']`` will be ``False``.
        """

        width, height, pixels, info = self.asDirect()
        if info['alpha']:
            raise Error("will not convert image with alpha channel to RGB")
        if not info['greyscale']:
            return width, height, pixels, info
        info['greyscale'] = False
        info['planes'] = 3

        if info['bitdepth'] > 8:
            def newarray():
                return array('H', [0])
        else:
            def newarray():
                return bytearray([0])

        def iterrgb():
            for row in pixels:
                a = newarray() * 3 * width
                for i in range(3):
                    a[i::3] = row
                yield a
        return width, height, iterrgb(), info

    def asRGBA(self):
        """
        Return image as RGBA pixels.
        Greyscales are expanded into RGB triplets;
        an alpha channel is synthesized if necessary.
        The return values are as for the :meth:`read` method except that
        the *info* reflect the returned pixels, not the source image.
        In particular, for this method
        ``info['greyscale']`` will be ``False``, and
        ``info['alpha']`` will be ``True``.
        """

        width, height, pixels, info = self.asDirect()
        if info['alpha'] and not info['greyscale']:
            return width, height, pixels, info
        typecode = 'BH'[info['bitdepth'] > 8]
        maxval = 2**info['bitdepth'] - 1
        maxbuffer = struct.pack('=' + typecode, maxval) * 4 * width

        if info['bitdepth'] > 8:
            def newarray():
                return array('H', maxbuffer)
        else:
            def newarray():
                return bytearray(maxbuffer)

        if info['alpha'] and info['greyscale']:
            # LA to RGBA
            def convert():
                for row in pixels:
                    # Create a fresh target row, then copy L channel
                    # into first three target channels, and A channel
                    # into fourth channel.
                    a = newarray()
                    convert_la_to_rgba(row, a)
                    yield a
        elif info['greyscale']:
            # L to RGBA
            def convert():
                for row in pixels:
                    a = newarray()
                    convert_l_to_rgba(row, a)
                    yield a
        else:
            assert not info['alpha'] and not info['greyscale']
            # RGB to RGBA

            def convert():
                for row in pixels:
                    a = newarray()
                    convert_rgb_to_rgba(row, a)
                    yield a
        info['alpha'] = True
        info['greyscale'] = False
        info['planes'] = 4
        return width, height, convert(), info


def decompress(data_blocks):
    """
    `data_blocks` should be an iterable that
    yields the compressed data (from the ``IDAT`` chunks).
    This yields decompressed byte strings.
    """

    # Currently, with no max_length parameter to decompress,
    # this routine will do one yield per IDAT chunk: Not very
    # incremental.
    d = zlib.decompressobj()
    # Each IDAT chunk is passed to the decompressor, then any
    # remaining state is decompressed out.
    for data in data_blocks:
        # :todo: add a max_length argument here to limit output size.
        yield bytearray(d.decompress(data))
    yield bytearray(d.flush())


def check_bitdepth_colortype(bitdepth, colortype):
    """
    Check that `bitdepth` and `colortype` are both valid,
    and specified in a valid combination.
    Returns (None) if valid, raise an Exception if not valid.
    """

    if bitdepth not in (1, 2, 4, 8, 16):
        raise FormatError("invalid bit depth %d" % bitdepth)
    if colortype not in (0, 2, 3, 4, 6):
        raise FormatError("invalid colour type %d" % colortype)
    # Check indexed (palettized) images have 8 or fewer bits
    # per pixel; check only indexed or greyscale images have
    # fewer than 8 bits per pixel.
    if colortype & 1 and bitdepth > 8:
        raise FormatError(
            "Indexed images (colour type %d) cannot"
            " have bitdepth > 8 (bit depth %d)."
            " See http://www.w3.org/TR/2003/REC-PNG-20031110/#table111 ."
            % (bitdepth, colortype))
    if bitdepth < 8 and colortype not in (0, 3):
        raise FormatError(
            "Illegal combination of bit depth (%d)"
            " and colour type (%d)."
            " See http://www.w3.org/TR/2003/REC-PNG-20031110/#table111 ."
            % (bitdepth, colortype))


def is_natural(x):
    """A non-negative integer."""
    try:
        is_integer = int(x) == x
    except (TypeError, ValueError):
        return False
    return is_integer and x >= 0


def undo_filter_sub(filter_unit, scanline, previous, result):
    """Undo sub filter."""

    ai = 0
    # Loops starts at index fu.  Observe that the initial part
    # of the result is already filled in correctly with
    # scanline.
    for i in range(filter_unit, len(result)):
        x = scanline[i]
        a = result[ai]
        result[i] = (x + a) & 0xff
        ai += 1


def undo_filter_up(filter_unit, scanline, previous, result):
    """Undo up filter."""

    for i in range(len(result)):
        x = scanline[i]
        b = previous[i]
        result[i] = (x + b) & 0xff


def undo_filter_average(filter_unit, scanline, previous, result):
    """Undo up filter."""

    ai = -filter_unit
    for i in range(len(result)):
        x = scanline[i]
        if ai < 0:
            a = 0
        else:
            a = result[ai]
        b = previous[i]
        result[i] = (x + ((a + b) >> 1)) & 0xff
        ai += 1


def undo_filter_paeth(filter_unit, scanline, previous, result):
    """Undo Paeth filter."""

    # Also used for ci.
    ai = -filter_unit
    for i in range(len(result)):
        x = scanline[i]
        if ai < 0:
            a = c = 0
        else:
            a = result[ai]
            c = previous[ai]
        b = previous[i]
        p = a + b - c
        pa = abs(p - a)
        pb = abs(p - b)
        pc = abs(p - c)
        if pa <= pb and pa <= pc:
            pr = a
        elif pb <= pc:
            pr = b
        else:
            pr = c
        result[i] = (x + pr) & 0xff
        ai += 1


def convert_la_to_rgba(row, result):
    for i in range(3):
        result[i::4] = row[0::2]
    result[3::4] = row[1::2]


def convert_l_to_rgba(row, result):
    """
    Convert a grayscale image to RGBA.
    This method assumes the alpha channel in result is
    already correctly initialized.
    """
    for i in range(3):
        result[i::4] = row


def convert_rgb_to_rgba(row, result):
    """
    Convert an RGB image to RGBA.
    This method assumes the alpha channel in result is
    already correctly initialized.
    """
    for i in range(3):
        result[i::4] = row[i::3]


# ---------------------------------------------------------


# Make the codec discoverable
def udsp_get_image_codec():
    return PNGCodec
