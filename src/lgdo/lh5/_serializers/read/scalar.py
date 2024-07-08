from __future__ import annotations

import logging

import numpy as np

from ....types import Scalar
from ...exceptions import LH5DecodeError

log = logging.getLogger(__name__)


def _h5_read_scalar(
    h5d,
    obj_buf=None,
):
    value = h5d[()]
    attrs = dict(h5d.attrs)

    # special handling for bools
    # (c and Julia store as uint8 so cast to bool)
    if attrs["datatype"] == "bool":
        value = np.bool_(value)

    if obj_buf is not None:
        if not isinstance(obj_buf, Scalar):
            msg = "object buffer a Scalar"
            raise LH5DecodeError(msg, h5d)

        obj_buf.value = value
        obj_buf.attrs.update(attrs)
        return obj_buf, 1

    return Scalar(value=value, attrs=attrs), 1
