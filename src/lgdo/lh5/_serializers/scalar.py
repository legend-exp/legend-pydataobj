from __future__ import annotations

import logging

import numpy as np

from ...types import Scalar

log = logging.getLogger(__name__)


def _h5_read_scalar(
    name,
    h5f,
    obj_buf=None,
):
    value = h5f[name][()]

    # special handling for bools
    # (c and Julia store as uint8 so cast to bool)
    if h5f[name].attrs["datatype"] == "bool":
        value = np.bool_(value)

    if obj_buf is not None:
        obj_buf.value = value
        obj_buf.attrs.update(h5f[name].attrs)
        return obj_buf, 1

    return Scalar(value=value, attrs=h5f[name].attrs), 1
