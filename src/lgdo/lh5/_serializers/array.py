from __future__ import annotations

import logging
import sys

from ...types import Array, ArrayOfEqualSizedArrays, FixedSizeArray
from .ndarray import _h5_read_ndarray

log = logging.getLogger(__name__)


def _h5_read_array(
    name,
    h5f,
    start_row=0,
    n_rows=sys.maxsize,
    idx=None,
    use_h5idx=False,
    obj_buf=None,
    obj_buf_start=0,
):
    nda, attrs, n_rows_to_read = _h5_read_ndarray(
        name,
        h5f,
        start_row=start_row,
        n_rows=n_rows,
        idx=idx,
        use_h5idx=use_h5idx,
        obj_buf=obj_buf,
        obj_buf_start=obj_buf_start,
    )

    if obj_buf is None:
        return Array(nda=nda, attrs=attrs), n_rows_to_read

    if set(obj_buf.attrs.keys()) != set(attrs.keys()):
        msg = (
            f"attrs mismatch. "
            f"obj_buf.attrs: {obj_buf.attrs}, "
            f"h5f[{name}].attrs: {attrs}"
        )
        raise RuntimeError(msg)

    return obj_buf, n_rows_to_read


def _h5_read_fixedsize_array(
    name,
    h5f,
    start_row=0,
    n_rows=sys.maxsize,
    idx=None,
    use_h5idx=False,
    obj_buf=None,
    obj_buf_start=0,
):
    nda, attrs, n_rows_to_read = _h5_read_ndarray(
        name,
        h5f,
        start_row=start_row,
        n_rows=n_rows,
        idx=idx,
        use_h5idx=use_h5idx,
        obj_buf=obj_buf,
        obj_buf_start=obj_buf_start,
    )

    if obj_buf is None:
        return FixedSizeArray(nda=nda, attrs=attrs), n_rows_to_read

    if set(obj_buf.attrs.keys()) != set(attrs.keys()):
        msg = (
            f"attrs mismatch. "
            f"obj_buf.attrs: {obj_buf.attrs}, "
            f"h5f[{name}].attrs: {attrs}"
        )
        raise RuntimeError(msg)

    return obj_buf, n_rows_to_read


def _h5_read_array_of_equalsized_arrays(
    name,
    h5f,
    start_row=0,
    n_rows=sys.maxsize,
    idx=None,
    use_h5idx=False,
    obj_buf=None,
    obj_buf_start=0,
):
    nda, attrs, n_rows_to_read = _h5_read_ndarray(
        name,
        h5f,
        start_row=start_row,
        n_rows=n_rows,
        idx=idx,
        use_h5idx=use_h5idx,
        obj_buf=obj_buf,
        obj_buf_start=obj_buf_start,
    )

    if obj_buf is None:
        return (
            ArrayOfEqualSizedArrays(nda=nda, attrs=attrs),
            n_rows_to_read,
        )

    if set(obj_buf.attrs.keys()) != set(attrs.keys()):
        msg = (
            f"attrs mismatch. "
            f"obj_buf.attrs: {obj_buf.attrs}, "
            f"h5f[{name}].attrs: {attrs}"
        )
        raise RuntimeError(msg)

    return obj_buf, n_rows_to_read
