"""
Codecs module

"""

import os
import importlib
import warnings

from . import const as _cst


class CodecRegistry(object):
    """
    This class implements functionality to manage codecs
    operations, such as loading, registering, checking, etc.

    Attributes
    ----------
    _registry: dict
        The codecs registry. Each record in the registry is
        keyed with the media format handled by the codec and
        has a structure with at least the following fields:

        "<format>" => {
                        "description": "<codec description>"
                        "module": "<codec module locator>"
                        "codec": <codec implementation class>
                      }

    _mediatype: str
        A string indicating the media type to be registered
        (it is assumed that the codecs are located in a dir
        with the same name as this field)

    Notes
    -----
    This class should be used from the media package's top
    level dir (where the __init__.py is) as it assumes that
    codecs are located in directories at the top level of
    the package.

    TODO: should probably be called CodecManager

    """
    def __init__(self, media):
        self._registry = {}
        self._mediatype = media

    def __getitem__(self, fmt):
        """
        Get the codec for a given media format

        Parameters
        ----------
        fmt: str
            The codec format

        Returns
        -------
        dict
            A codec record

        """
        return self._registry[fmt]

    def __iter__(self):
        """
        Get an iterator over the registered codecs

        Returns
        -------
        iterator

        """
        return iter(self._registry.items())

    def get_formats(self):
        """
        Get the registered media formats

        Returns
        -------
        list[str]

        """
        return list(self._registry.keys())

    def load(self):
        """
        Load and register all available codecs

        Parameters
        ----------

        Returns
        -------
        CodecRegistry

        """
        def is_codec_file(f):
            return (os.path.isfile(f[0]) and
                    f[2].lower() == ".py" and
                    f[1].endswith(_cst.CODEC_FILE_SUFFIX))

        if not self._mediatype:
            raise ValueError("Undefined media type")

        if len(self._registry):
            self._registry.clear()

        mtype = self._mediatype
        media_dir = os.path.dirname(os.path.realpath(__file__))
        codecs_dir = os.path.sep.join([media_dir, mtype])

        for file in os.listdir(codecs_dir):
            filepath = os.path.sep.join([codecs_dir, file])
            fname, fext = os.path.splitext(file)
            if is_codec_file([filepath, fname, fext]):
                mod_loc = ".".join([__package__, mtype, fname])
                mod = importlib.import_module(mod_loc)
                try:
                    get_codec = getattr(mod, _cst.CODEC_GET_FUNCTION)
                except AttributeError:
                    # Not a valid codec
                    warnings.warn(
                        "\nAttempt to load codec '{}' from the '{}' package "
                        "that does not implement '{}()'. "
                        "All codecs must implement the above mentioned "
                        "function at the module level in order to be correctly "
                        "discovered and registered."
                        .format(file, mtype, _cst.CODEC_GET_FUNCTION))
                    # TODO: we should delete invalid loaded codec modules
                else:
                    # Register the codec
                    codec = get_codec()
                    codec.module = mod_loc
                    self._register(codec)
        return self

    def _register(self, codec):
        """
        Register a codec for a specific media format

        Parameters
        ----------
        codec: class
            The class implementing the codec

        Returns
        -------
        None

        """
        try:
            self._registry[codec.format] = {
                "description": codec.description,
                "module": codec.module,
                "codec": codec
            }
        except (KeyError, AttributeError):
            raise AttributeError(
                "Codecs must define a 'format' and 'description' attribute"
            )
