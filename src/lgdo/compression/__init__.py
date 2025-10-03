r"""Data compression utilities.

This subpackage collects all LEGEND custom data compression (encoding) and
decompression (decoding) algorithms.

Available lossless waveform compression algorithms:

* :class:`.RadwareSigcompress`, a Python port of the C algorithm
  `radware-sigcompress` by D. Radford.
* :class:`.ULEB128ZigZagDiff` variable-length base-128 encoding of waveform
  differences.

All waveform compression algorithms inherit from the :class:`.WaveformCodec`
abstract class.

:func:`~.generic.encode` and :func:`~.generic.decode` provide a high-level
interface for encoding/decoding :class:`~.lgdo.LGDO`\ s.

>>> from lgdo import WaveformTable, compression
>>> wftbl = WaveformTable(...)
>>> enc_wft = compression.encode(wftable, RadwareSigcompress(codec_shift=-23768)
>>> compression.decode(enc_wft) # == wftbl
"""

from __future__ import annotations

from importlib import import_module

from .base import WaveformCodec
from .generic import decode, encode

# mapping from codec class to module for lazy loading
_codec_classes = {
    "RadwareSigcompress": "radware",
    "ULEB128ZigZagDiff": "varlen",
}

__all__ = list(_codec_classes)
__all__ += [
    "WaveformCodec",
    "decode",
    "encode",
]


# Lazy loader
def __getattr__(name):
    if name in _codec_classes:
        mod_name = _codec_classes[name]
        mod = import_module(f".{mod_name}", __name__)
        codec = getattr(mod, name)
        globals().update({name: codec})
        return codec
    msg = f"module {__name__} has no attribute {name}"
    raise AttributeError(msg)


def __dir__():
    return __all__ + list(globals().keys())
