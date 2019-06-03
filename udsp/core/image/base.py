"""
Image module

"""


class Metadata(object):
    """
    Image metadata container

    Attributes
    ----------
    size: tuple
        The size of the image, as in (width, height)
    bps: int
        The bits per sample (not pixel, beware)
    planes: int
        The number of colour planes

    Note that these are the minimum set of metadata
    required to identify an image. Specific formats may
    add more.

    """
    def __init__(self):
        self.size = None
        self.bps = None
        self.planes = None


class Image(object):
    """
    Abstract base class for image codecs

    """

    format = None
    description = None

    def __init__(self, stream):
        """
        Create a new image from a given stream

        Parameters
        ----------
        stream
            Any stream implementing an open/read/write/close interface

        """
        pass

    def load(self):
        """
        Load and returns the image data

        Returns
        -------
        list[]
            A list of matrices representing the image's colour planes

        """
        raise NotImplementedError

    def save(self):
        """
        Save the image

        Returns
        -------
        None

        """
        raise NotImplementedError

    @property
    def metadata(self):
        """
        Get metadata for the image

        Returns
        -------
        Metadata

        """
        raise NotImplementedError
