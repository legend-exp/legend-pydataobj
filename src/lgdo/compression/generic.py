# ruff: noqa: UP007
from __future__ import annotations

import logging
from typing import Optional, Union

from .. import types as lgdo
from . import radware, varlen
from .base import WaveformCodec

log = logging.getLogger(__name__)


def encode(
    obj: Union[lgdo.VectorOfVectors, lgdo.ArrayOfEqualsizedArrays],
    codec: Optional[Union[WaveformCodec, str]] = None,
) -> Union[lgdo.VectorOfEncodedVectors, lgdo.ArrayOfEncodedEqualSizedArrays]:
    """Encode LGDOs with `codec`.

    Defines behaviors for each implemented waveform encoding algorithm.

    Parameters
    ----------
    obj
        LGDO array type.
    codec
        algorithm to be used for encoding.
    """
    log.debug(f"encoding {obj!r} with {codec}")

    if _is_codec(codec, radware.RadwareSigcompress):
        enc_obj = radware.encode(obj, shift=codec.codec_shift)
    elif _is_codec(codec, varlen.ULEB128ZigZagDiff):
        enc_obj = varlen.encode(obj)
    else:
        msg = f"'{codec}' not supported"
        raise ValueError(msg)

    enc_obj.attrs |= codec.asdict()

    return enc_obj


def decode(
    obj: Union[lgdo.VectorOfEncodedVectors, lgdo.ArrayOfEncodedEqualSizedArrays],
    out_buf: Optional[lgdo.ArrayOfEqualSizedArrays] = None,
) -> Union[lgdo.VectorOfVectors, lgdo.ArrayOfEqualsizedArrays]:
    """Decode encoded LGDOs.

    Defines decoding behaviors for each implemented waveform encoding
    algorithm. Expects to find the codec (and its parameters) the arrays where
    encoded with among the LGDO attributes.

    Parameters
    ----------
    obj
        LGDO array type.
    out_buf
        pre-allocated LGDO for the decoded signals. See documentation of
        wrapped encoders for limitations.
    """
    if "codec" not in obj.attrs:
        msg = (
            "object does not carry any 'codec' attribute, I don't know how to decode it"
        )
        raise RuntimeError(msg)

    codec = obj.attrs["codec"]
    log.debug(f"decoding {obj!r} with {codec}")

    if _is_codec(codec, radware.RadwareSigcompress):
        return radware.decode(
            obj, sig_out=out_buf, shift=int(obj.attrs.get("codec_shift", 0))
        )

    if _is_codec(codec, varlen.ULEB128ZigZagDiff):
        return varlen.decode(obj, sig_out=out_buf)

    msg = f"'{codec}' not supported"
    raise ValueError(msg)


def _is_codec(ident: Union[WaveformCodec, str], codec) -> bool:
    if isinstance(ident, WaveformCodec):
        return isinstance(ident, codec)

    if isinstance(ident, str):
        return ident == codec().codec

    msg = "input must be WaveformCodec object or string identifier"
    raise ValueError(msg)
