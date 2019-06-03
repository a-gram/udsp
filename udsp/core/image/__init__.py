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
3. A codec must implement the Image interface AND a 'getter'
   function with a specific name (see the _CODEC_GET_FUNCTION
   constant below) in order to be discovered and registered.
   This function must be defined at the module level and must
   return the class implementing the codec.

"""

import os
import importlib
import warnings

# Name of the codec getter function
_CODEC_GET_FUNCTION = "udsp_get_image_codec"

# Suffix for codec module files
_CODEC_FILE_SUFFIX = "_codec"

# Codecs "registry"
_supported_formats = {}


# ---------------------------------------------------------
#                       Public API
# ---------------------------------------------------------


def from_file(filename):
    """
    Factory method to create image objects from files

    Parameters
    ----------
    filename: str
        The full path to the image file

    Returns
    -------
    Image

    """
    fmt = os.path.splitext(filename)[1].replace(".", "")

    try:
        image = _supported_formats[fmt]
    except KeyError:
        raise RuntimeError("Unsupported image format: %s" % fmt)

    return image["codec"](filename)


def get_supported_formats():
    """
    Get a list of supported image codecs

    Returns
    -------
    list

    """
    return list(_supported_formats.keys())


def print_supported_formats():
    """
    Print a list of supported image codecs to stdout

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
#                     Private members
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
