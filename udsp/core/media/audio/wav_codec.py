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
    description = "WAV audio decoder/encoder"

    # Sample format to array type mapping
    _FMT_ACODE = {
        8:  "B",
        16: "h",
        24: "i",
        32: "i"
    }

    def __init__(self, rstream=None, wstream=None):

        super().__init__(rstream, wstream)
        self._reader = wave.open(rstream, "rb") if rstream else None
        self._writer = wave.open(wstream, "wb") if wstream else None

        # In write mode, if no data is written exceptions will
        # raise when the file is closed, so give some defaults.
        if wstream:
            self._writer.setsampwidth(2)
            self._writer.setnchannels(1)
            self._writer.setframerate(44100)

    def decode(self):

        self._check_stream("read")

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

    def encode(self, data, meta):

        self._check_stream("write")
        self.set_metadata(meta)

        nframes = meta.size
        bsamples = BLOCKSIZE * meta.channels
        rsamples = 0

        while nframes > 0:
            bdata = data[rsamples: rsamples + bsamples]
            self._writer.writeframes(bdata.tobytes())
            nframes -= (len(bdata) / meta.channels)
            rsamples += len(bdata)

        # bdata = data[rsamples: rsamples + bsamples]
        # while len(bdata):
        #     self._writer.writeframes(bdata.tobytes())
        #     rsamples += len(bdata)
        #     bdata = data[rsamples: rsamples + bsamples]

    def get_metadata(self):

        self._check_stream("read")

        meta = Metadata()
        meta.size = self._reader.getnframes()
        meta.bps = self._reader.getsampwidth() * 8
        meta.channels = self._reader.getnchannels()
        meta.resolution = self._reader.getframerate()
        return meta

    def set_metadata(self, meta):

        self._check_stream("write")

        self._writer.setnframes(meta.size)
        self._writer.setsampwidth(int(meta.bps / 8))
        self._writer.setnchannels(meta.channels)
        self._writer.setframerate(meta.resolution)

    def _check_stream(self, mode):

        if mode == "read":
            if not self._reader:
                raise RuntimeError("Codec not set in read mode")
        elif mode == "write":
            if not self._writer:
                raise RuntimeError("Codec not set in write mode")
        else:
            raise RuntimeError("Bug")


# Make the codec discoverable
def udsp_get_media_codec():
    return WAVCodec
