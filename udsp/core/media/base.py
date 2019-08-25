"""
Multimedia base classes module

"""

import os


class Metadata(object):
    """
    Media metadata container

    Attributes
    ----------
    size: int, tuple
        The size of the media
    bps: int
        The bits per sample
    channels: int
        The number of channels
    resolution: int, tuple
        The data resolution

    Notes
    -----
    The fields defined here are the minimum set of metadata
    required to identify a media object. Specific formats may
    have more.

    """
    def __init__(self, size=None,
                       bps=None,
                       channels=None,
                       resolution=None):
        self.size = size
        self.bps = bps
        self.channels = channels
        self.resolution = resolution

    def __str__(self):
        return ("\nsize:       {}\n"
                "bps:        {}\n"
                "channels:   {}\n"
                "resolution: {}\n"
                .format(
                 self.size,
                 self.bps,
                 self.channels,
                 self.resolution))


class MediaCodec(object):
    """
    Abstract base class for multimedia codecs

    """

    format = None
    description = None

    def __init__(self, rstream, wstream):
        """
        Create a media codec from a given stream

        Parameters
        ----------
        rstream
            Any stream implementing an open/read/close interface,
            or a stream identifier (e.g. string, int)
        wstream
            Any stream implementing an open/write/close interface,
            or a stream identifier (e.g. string, int)

        """
        pass

    def decode(self):
        """
        Decode and return the raw media data

        Returns
        -------
        array
            An array with the raw data samples

        """
        raise NotImplementedError

    def encode(self, data, meta):
        """
        Encode and save the media data

        Parameters
        ----------
        data: array
            An array with the raw data samples

        Returns
        -------
        None

        """
        raise NotImplementedError

    def get_metadata(self):
        """
        Get the media metadata

        Returns
        -------
        Metadata

        """
        raise NotImplementedError

    def set_metadata(self, meta):
        """
        Set the media metadata

        Parameters
        ----------
        meta: Metadata

        Returns
        -------

        """
        raise NotImplementedError


class MediaObject(object):
    """
    Abstract base class for multimedia objects

    Attributes
    ----------
    _codec: MediaCodec
        A codec object to read/write the media data
    _meta: Metadata
        The media metadata
    _mediatype: str
        The type of media (image, audio, etc.)
    _codecs: CodecRegistry
        The registry to be accessed to get the codecs

    """
    _mediatype = None  # Abstract
    _codecs = None     # Abstract

    def __init__(self):
        self._meta = None   # Abstract
        self._codec = None  # Abstract

    @property
    def metadata(self):
        """
        Get the media metadata

        Returns
        -------
        Metadata

        """
        if self._meta is None:
            self._meta = self._codec.get_metadata()
        return self._meta

    @property
    def mediatype(self):
        """
        Get the media type

        Returns
        -------
        str

        """
        return self._mediatype

    @property
    def format(self):
        """
        Get the media format

        Returns
        -------
        str

        """
        return self._codec.format

    @property
    def description(self):
        """
        Get the media description

        Returns
        -------
        str

        """
        return self._codec.description

    def load(self):
        """
        Load and return the media data

        Returns
        -------
        list[]
            A list of vectors/matrices representing the media channels
            (i.e. colour planes, audio channels, etc.) in the format that
            is understood by the signal's functions.

        """
        raise NotImplementedError

    def save(self, data, meta):
        """
        Save the media data

        Parameters
        ----------
        data: list[]
            A list of vectors/matrices representing the media channels
        meta: Metadata
            The media metadata

        Returns
        -------

        """
        raise NotImplementedError

    @classmethod
    def from_file(cls, filename):
        """
        Create media objects from files

        Parameters
        ----------
        cls: class
            A media class
        filename: str
            The full path to the media file

        Returns
        -------
        MediaObject

        """
        formatt = cls._get_codec(cls, filename)
        media = cls()
        media._codec = formatt["codec"](rstream=filename)
        return media

    @classmethod
    def to_file(cls, filename):
        """
        Create media objects to files

        Parameters
        ----------
        cls: class
            A media class
        filename: str
            The full path to the media file

        Returns
        -------
        MediaObject

        """
        formatt = cls._get_codec(cls, filename)
        media = cls()
        media._codec = formatt["codec"](wstream=filename)
        return media

    @classmethod
    def get_supported_formats(cls):
        """
        Get a list of supported media formats for a media class

        Returns
        -------
        list[str]

        """
        if cls is MediaObject:
            raise TypeError(
                "This method can't be called from the base class"
            )
        return cls._codecs.get_formats()

    @classmethod
    def print_supported_formats(cls):
        """
        Print a list of supported media formats for a media class

        Returns
        -------
        None

        """
        if cls is MediaObject:
            raise TypeError(
                "This method can't be called from the base class"
            )
        print("\nSupported {} formats"
              "\n-----------------------".format(cls._mediatype))
        if len(cls._codecs.get_formats()):
            for fmt, info in cls._codecs:
                print("{} - {}".format(fmt, info["description"]))
        else:
            print("No codecs found")
        print("")

    @classmethod
    def _use_codecs(cls, reg):
        """
        Give the media class access to the specified codec registry

        This shall be considered a package-level protected method
        to be used in an internal initialization module/function
        to set the codec registry that a media class should access
        when creating instances. It must be implemented in concrete
        classes since different media may use different registries.

        Parameters
        ----------
        reg: CodecRegistry

        Returns
        -------

        """
        raise NotImplementedError

    @staticmethod
    def _get_codec(cls, filename):
        """
        Look up the codec for a given media file

        Parameters
        ----------
        cls
        filename

        Returns
        -------

        """
        if cls is MediaObject:
            raise TypeError(
                "This method can't be called from the base class"
            )
        if not cls._codecs:
            raise ValueError("The codecs registry is not set")

        ext = os.path.splitext(filename)[1].replace(".", "")

        try:
            codec = cls._codecs[ext]
        except KeyError:
            raise RuntimeError("Unsupported {} format: {}"
                               .format(cls._mediatype, ext))
        return codec
