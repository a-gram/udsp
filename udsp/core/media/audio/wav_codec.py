"""
WAV codec implementation

"""

import wave

from array import array
from struct import unpack_from, pack
from ..base import MediaCodec, Metadata

BLOCKSIZE = 8192


class WAVCodec(MediaCodec):

    format = "wav"
    description = "WAV audio coder/encoder"

    # Sample format to array type mapping
    _FMT_ACODE = {
        8:  "B",
        16: "h",
        24: "i",
        32: "i"
    }

    def __init__(self, stream):
        super().__init__(stream)
        self._reader = wave.open(stream, "rb")
        self._writer = None

    def decode(self):

        nframes = self._reader.getnframes()
        Bps = self._reader.getsampwidth()
        nchans = self._reader.getnchannels()
        bps = Bps * 8  # bits per sample
        Bpf = Bps * nchans  # bytes per frame
        atype = self._FMT_ACODE[bps]
        samples = array(atype)

        # def unpack24(b):
        #     ib = bytearray(4)
        #     for i in range(0, len(b), 3):
        #         ib[1:4] = b[i:i + 3]
        #         yield unpack_from("i", ib, 0)[0]

        def unpack24(b):
            for i in range(0, len(b), 3):
                yield int.from_bytes(b[i:i + 3],
                                     "little", signed=True)

        while nframes > 0:
            fbytes = self._reader.readframes(BLOCKSIZE)
            if bps == 24:
                samples.extend(array(atype, unpack24(fbytes)))
            else:
                samples.frombytes(fbytes)
            nframes -= (len(fbytes) / Bpf)
        return samples

    def encode(self):
        # TODO: Not implemented
        pass

    def get_metadata(self):
        meta = Metadata()
        meta.size = self._reader.getnframes()
        meta.bps = self._reader.getsampwidth() * 8
        meta.channels = self._reader.getnchannels()
        meta.resolution = self._reader.getframerate()
        return meta


# Make the codec discoverable
def udsp_get_media_codec():
    return WAVCodec
