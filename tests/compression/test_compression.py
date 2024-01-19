from __future__ import annotations

from lgdo import ArrayOfEncodedEqualSizedArrays, compression
from lgdo.compression import RadwareSigcompress
from lgdo.compression.generic import _is_codec


def test_encode_decode_array(wftable):
    result = compression.encode(
        wftable.values, codec=RadwareSigcompress(codec_shift=-32768)
    )
    assert isinstance(result, ArrayOfEncodedEqualSizedArrays)
    assert len(result) == len(wftable)
    assert result.attrs["codec"] == "radware_sigcompress"
    assert result.attrs["codec_shift"] == -32768

    compression.decode(result)


def test_is_codec():
    assert _is_codec("radware_sigcompress", RadwareSigcompress)
