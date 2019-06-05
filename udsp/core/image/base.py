"""
Image codec module

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

    Notes
    -----
    The fields defined here are the minimum set of metadata
    required to identify an image. Specific formats may
    add more.

    """
    def __init__(self):
        self.size = None
        self.bps = None
        self.planes = None


class ImageCodec(object):
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

    def decode(self):
        """
        Load and returns the image pixels data

        Returns
        -------
        list[]
            A list of matrices representing the image colour planes

        """
        raise NotImplementedError

    def encode(self):
        """
        Save the image data

        Returns
        -------
        None

        """
        raise NotImplementedError

    def get_metadata(self):
        """
        Get the image metadata

        Returns
        -------
        Metadata

        """
        raise NotImplementedError
