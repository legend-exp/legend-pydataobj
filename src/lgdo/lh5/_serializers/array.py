from __future__ import annotations

import logging

from ...types import Array, ArrayOfEqualSizedArrays, FixedSizeArray
from .ndarray import _h5_read_ndarray

log = logging.getLogger(__name__)


def _h5_read_array_generic(type_, name, h5f, **kwargs):
    nda, attrs, n_rows_to_read = _h5_read_ndarray(name, h5f, **kwargs)

    obj_buf = kwargs["obj_buf"]

    if obj_buf is None:
        return type_(nda=nda, attrs=attrs), n_rows_to_read

    check_obj_buf_attrs(obj_buf.attrs, attrs, f"{h5f.filename}[{name}]")

    return obj_buf, n_rows_to_read


def _h5_read_array(name, h5f, **kwargs):
    return _h5_read_array_generic(Array, name, h5f, **kwargs)


def _h5_read_fixedsize_array(name, h5f, **kwargs):
    return _h5_read_array_generic(FixedSizeArray, name, h5f, **kwargs)


def _h5_read_array_of_equalsized_arrays(name, h5f, **kwargs):
    return _h5_read_array_generic(ArrayOfEqualSizedArrays, name, h5f, **kwargs)


def check_obj_buf_attrs(attrs, new_attrs, name):
    if attrs != new_attrs:
        msg = (
            f"existing LGDO buffer and new data chunk have different attributes: "
            f"obj_buf.attrs={attrs} != {name}.attrs={new_attrs}"
        )
        raise RuntimeError(msg)
