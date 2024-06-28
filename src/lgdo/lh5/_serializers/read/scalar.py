from __future__ import annotations

import logging

import numpy as np

from ....types import Scalar
from ...exceptions import LH5DecodeError

log = logging.getLogger(__name__)


def _h5_read_scalar(
    name,
    h5f,
    obj_buf=None,
    metadata=None,  # specific to requested object
):
    value = h5f[name][()]

    frommeta = False
    if metadata is not None:
        frommeta = metadata["attrs"]["datatype"] == "bool"

    # special handling for bools
    # (c and Julia store as uint8 so cast to bool)
    if frommeta or h5f[name].attrs["datatype"] == "bool":
        value = np.bool_(value)

    if obj_buf is not None:
        if not isinstance(obj_buf, Scalar):
            msg = "object buffer a Scalar"
            raise LH5DecodeError(msg, h5f, name)

        obj_buf.value = value
        if frommeta:
            obj_buf.attrs.update(metadata["attrs"])
        else:
            obj_buf.attrs.update(h5f[name].attrs)
        return obj_buf, 1

    if frommeta:
        thisattrs = metadata["attrs"]
    else:
        thisattrs = h5f[name].attrs
    return Scalar(value=value, attrs=thisattrs), 1
