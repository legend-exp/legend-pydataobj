from __future__ import annotations

import logging

from ....types import Array, ArrayOfEqualSizedArrays, FixedSizeArray
from . import utils
from .ndarray import _h5_read_ndarray

log = logging.getLogger(__name__)


def _h5_read_array_generic(type_, h5d, **kwargs):
    nda, attrs, n_rows_to_read = _h5_read_ndarray(h5d, **kwargs)

    obj_buf = kwargs["obj_buf"]

    if obj_buf is None:
        return type_(nda=nda, attrs=attrs), n_rows_to_read

    utils.check_obj_buf_attrs(obj_buf.attrs, attrs, h5d)

    return obj_buf, n_rows_to_read


def _h5_read_array(h5d, **kwargs):
    return _h5_read_array_generic(Array, h5d, **kwargs)


def _h5_read_fixedsize_array(h5d, **kwargs):
    return _h5_read_array_generic(FixedSizeArray, h5d, **kwargs)


def _h5_read_array_of_equalsized_arrays(h5d, **kwargs):
    return _h5_read_array_generic(ArrayOfEqualSizedArrays, h5d, **kwargs)
