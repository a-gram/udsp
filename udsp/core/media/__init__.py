"""
Multimedia module

This module discovers and registers all available media codecs
found in the 'media' package. A codec is a reader/writer object
that can understand and manipulate media in a specific format.
The following protocol must be observed in order to add support
for specific image formats:

1. A codec must be implemented as a single file/module located
   in the appropriate directory within the media package.
2. A codec file is identified by the name ending with a specfic
   suffix (see CODEC_FILE_SUFFIX constant in 'const.py')
3. A codec must implement the MediaCodec interface AND a 'getter'
   function with a specific signature (see the CODEC_GET_FUNCTION
   constant in 'const.py') in order to be discovered and registered.
   This function must be defined at the module level and must
   return the class implementing the codec.

"""

import itertools as _itools

from .base import MediaObject, MediaCodec, Metadata
from .codecs import CodecRegistry
from . import const as _cst
from .. import mtx as _mtx


# ---------------------------------------------------------
#                       Public API
# ---------------------------------------------------------


class Image(MediaObject):
    """
    Image object

    """

    _mediatype = "image"
    _codecs = None

    def __init__(self):
        super().__init__()

    def load(self):
        data = self._codec.decode()
        return self._to_mat(data)

    def save(self):
        # TODO: Not implemented
        pass

    @classmethod
    def _use_codecs(cls, reg):
        cls._codecs = reg

    def _to_mat(self, data):
        """
        Convert the media data into matrices

        """
        nchans = self.metadata.channels
        channels = [_mtx.vec_new(0) for _ in range(nchans)]

        for sline in data:
            for c in range(nchans):
                it = _itools.islice(sline, c, None, nchans)
                row = _mtx.vec_new(0, it)
                channels[c].append(row)
        return channels


class Audio(MediaObject):
    """
    Audio object

    """

    _mediatype = "audio"
    _codecs = None

    def __init__(self):
        super().__init__()

    def load(self):
        data = self._codec.decode()
        return self._to_mat(data)

    def save(self):
        # TODO: Not implemented
        pass

    @classmethod
    def _use_codecs(cls, reg):
        cls._codecs = reg

    def _to_mat(self, data):
        """
        Convert the media data into matrices

        """
        nchans = self.metadata.channels
        channels = []

        for c in range(nchans):
            it = _itools.islice(data, c, None, nchans)
            chan = _mtx.vec_new(0, it)
            channels.append(chan)
        return channels


# ---------------------------------------------------------
#                  Module initialization
# ---------------------------------------------------------


# Create the media registries and load the codecs
_img_codecs = CodecRegistry(_cst.CODEC_DIRNAME_IMAGE)
_aud_codecs = CodecRegistry(_cst.CODEC_DIRNAME_AUDIO)
_img_codecs.load()
_aud_codecs.load()

# Set the codec registries used by the media objects
Image._use_codecs(_img_codecs)
Audio._use_codecs(_aud_codecs)
