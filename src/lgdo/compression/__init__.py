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

__all__ = [
    "RadwareSigcompress",
    "ULEB128ZigZagDiff",
    "WaveformCodec",
    "decode",
    "encode",
]


def __getattr__(name):
    if name == "RadwareSigcompress":
        return import_module(".radware", __name__).RadwareSigcompress
    if name == "ULEB128ZigZagDiff":
        return import_module(".varlen", __name__).ULEB128ZigZagDiff
    module_name = __name__
    msg = "module {!r} has no attribute {!r}".format(module_name, name)  # noqa: UP032
    raise AttributeError(msg)
