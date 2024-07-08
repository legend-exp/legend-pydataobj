from __future__ import annotations

import logging
import sys

from .... import compression as compress
from ....types import (
    ArrayOfEncodedEqualSizedArrays,
    VectorOfEncodedVectors,
)
from ...exceptions import LH5DecodeError
from .array import (
    _h5_read_array,
)
from .scalar import _h5_read_scalar
from .vector_of_vectors import _h5_read_vector_of_vectors

log = logging.getLogger(__name__)


def _h5_read_array_of_encoded_equalsized_arrays(
    h5g,
    **kwargs,
):
    return _h5_read_encoded_array(ArrayOfEncodedEqualSizedArrays, h5g, **kwargs)


def _h5_read_vector_of_encoded_vectors(
    h5g,
    **kwargs,
):
    return _h5_read_encoded_array(VectorOfEncodedVectors, h5g, **kwargs)


def _h5_read_encoded_array(
    lgdotype,
    h5g,
    start_row=0,
    n_rows=sys.maxsize,
    idx=None,
    use_h5idx=False,
    obj_buf=None,
    obj_buf_start=0,
    decompress=True,
):
    if lgdotype not in (ArrayOfEncodedEqualSizedArrays, VectorOfEncodedVectors):
        msg = f"unsupported read of encoded type {lgdotype.__name__}"
        raise LH5DecodeError(msg, h5g)

    if not decompress and obj_buf is not None and not isinstance(obj_buf, lgdotype):
        msg = f"object buffer is not a {lgdotype.__name__}"
        raise LH5DecodeError(msg, h5g)

    # read out decoded_size, either a Scalar or an Array
    decoded_size_buf = encoded_data_buf = None
    if obj_buf is not None and not decompress:
        decoded_size_buf = obj_buf.decoded_size
        encoded_data_buf = obj_buf.encoded_data

    if lgdotype is VectorOfEncodedVectors:
        decoded_size, _ = _h5_read_array(
            h5g["decoded_size"],
            start_row=start_row,
            n_rows=n_rows,
            idx=idx,
            use_h5idx=use_h5idx,
            obj_buf=None if decompress else decoded_size_buf,
            obj_buf_start=0 if decompress else obj_buf_start,
        )

    else:
        decoded_size, _ = _h5_read_scalar(
            h5g["decoded_size"],
            obj_buf=None if decompress else decoded_size_buf,
        )

    # read out encoded_data, a VectorOfVectors
    encoded_data, n_rows_read = _h5_read_vector_of_vectors(
        h5g["encoded_data"],
        start_row=start_row,
        n_rows=n_rows,
        idx=idx,
        use_h5idx=use_h5idx,
        obj_buf=None if decompress else encoded_data_buf,
        obj_buf_start=0 if decompress else obj_buf_start,
    )

    # return the still encoded data in the buffer object, if there
    if obj_buf is not None and not decompress:
        return obj_buf, n_rows_read

    # otherwise re-create the encoded LGDO
    rawdata = lgdotype(
        encoded_data=encoded_data,
        decoded_size=decoded_size,
        attrs=dict(h5g.attrs),
    )

    # already return if no decompression is requested
    if not decompress:
        return rawdata, n_rows_read

    # if no buffer, decode and return
    if obj_buf is None and decompress:
        return compress.decode(rawdata), n_rows_read

    # eventually expand provided obj_buf, if too short
    buf_size = obj_buf_start + n_rows_read
    if len(obj_buf) < buf_size:
        obj_buf.resize(buf_size)

    # use the (decoded object type) buffer otherwise
    if lgdotype is ArrayOfEncodedEqualSizedArrays:
        compress.decode(rawdata, obj_buf[obj_buf_start:buf_size])

    elif lgdotype is VectorOfEncodedVectors:
        # FIXME: not a good idea. an in place decoding version
        # of decode would be needed to avoid extra memory
        # allocations
        for i, wf in enumerate(compress.decode(rawdata)):
            obj_buf[obj_buf_start + i] = wf

    return obj_buf, n_rows_read
