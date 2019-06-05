"""
Image module

This module discovers and registers all available image codecs
found in the image package. A 'codec' is a reader/writer object
that can understand and manipulate images in a specific format.
The following protocol must be observed in order to add support
for specific image formats:

1. A codec must be implemented as a single file/module located
   in the 'image' sub-package.
2. A codec file is identified by the name ending with a specfic
   suffix (see the _CODEC_FILE_SUFFIX constant below)
3. A codec must implement the ImageCodec interface AND a 'getter'
   function with a specific name (see the _CODEC_GET_FUNCTION
   constant below) in order to be discovered and registered.
   This function must be defined at the module level and must
   return the class implementing the codec.

"""

import os
import importlib
import warnings
import itertools

from .. import mtx as _mtx


# Name of the codec getter function
_CODEC_GET_FUNCTION = "udsp_get_image_codec"

# Suffix for codec module files
_CODEC_FILE_SUFFIX = "_codec"

# Codecs "registry"
_supported_formats = {}


# ---------------------------------------------------------
#                       Public API
# ---------------------------------------------------------


class Image(object):
    """
    Image object

    Attributes
    ----------
    _codec: ImageCodec

    _meta: Metadata

    """
    def __init__(self):
        self._meta = None
        self._codec = None

    @classmethod
    def from_file(cls, filename):
        """
        Create image objects from files

        Parameters
        ----------
        filename: str
            The full path to the image file

        Returns
        -------
        Image

        """
        ext = os.path.splitext(filename)[1].replace(".", "")

        try:
            formatt = _supported_formats[ext]
        except KeyError:
            raise RuntimeError("Unsupported image format: %s" % ext)

        image = cls()
        image._codec = formatt["codec"](filename)
        return image

    @property
    def metadata(self):
        if self._meta is None:
            self._meta = self._codec.get_metadata()
        return self._meta

    @property
    def format(self):
        return self._codec.format

    @property
    def description(self):
        return self._codec.description

    def load(self):
        data = self._codec.decode()
        return self._to_mat(data)

    def save(self):
        raise NotImplementedError

    def _to_mat(self, data):

        nplanes = self._meta.planes

        # Solution 1 (slower)
        #
        # nrows, ncols = img["size"][1], img["size"][0]
        # data = [*data]
        # planes = []
        #
        # def pixel(n, m):
        #     return data[n][m * nplanes + pixel.poff]
        # pixel.poff = 0
        #
        # for plane in range(nplanes):
        #     p = _mtx.mat_new(nrows, ncols, pixel)
        #     planes.append(p)
        #     pixel.poff += 1
        # return planes

        planes = [[] for _ in range(nplanes)]
        for sline in data:
            for p in range(nplanes):
                it = itertools.islice(sline, p, None, nplanes)
                row = _mtx.vec_new(0, it)
                planes[p].append(row)
        return planes

    @staticmethod
    def get_supported_formats():
        """
        Get a list of supported image formats

        Returns
        -------
        list[str]

        """
        return list(_supported_formats.keys())

    @staticmethod
    def print_supported_formats():
        """
        Print a list of supported image formats to stdout

        Returns
        -------
        None

        """
        for fmt, info in _supported_formats.items():
            print("\nSupported image formats")
            print("-----------------------")
            print("{} - {}".format(fmt, info["description"]))
        print("")


# ---------------------------------------------------------
#                  Module initialization
# ---------------------------------------------------------


def _register_codec(codec):
    """
    Register a codec for a specific image format

    Parameters
    ----------
    codec: class
        The class implementing the codec

    Returns
    -------
    None

    """
    try:
        _supported_formats[codec.format] = {
            "description": codec.description,
            "codec": codec
        }
    except KeyError:
        raise AttributeError(
            "Codecs must define a 'format' and 'description' attribute"
        )


def _load_codecs():
    """
    Load and register all available codecs

    Returns
    -------
    None

    """
    def is_codec_file(f):
        return (os.path.isfile(f[0]) and
                f[2].lower() == ".py" and
                f[1].endswith(_CODEC_FILE_SUFFIX))

    codecs_dir = os.path.dirname(os.path.realpath(__file__))

    for file in os.listdir(codecs_dir):
        filepath = os.sep.join([codecs_dir, file])
        fname, fext = os.path.splitext(file)
        if is_codec_file([filepath, fname, fext]):
            mod_loc = ".".join([__package__, fname])
            mod = importlib.import_module(mod_loc)
            try:
                get_codec = getattr(mod, _CODEC_GET_FUNCTION)
            except AttributeError:
                # Not a valid codec
                warnings.warn(
                    "\nAttempt to load codec '{}' in the 'image' package "
                    "that does not implement '{}()'. "
                    "All codecs must implement the above mentioned "
                    "function at the module level in order to be correctly "
                    "discovered and registered."
                    .format(file, _CODEC_GET_FUNCTION))
                # TODO: we should delete invalid loaded codec modules
            else:
                _register_codec(get_codec())


# Load and register all available codecs
_load_codecs()
