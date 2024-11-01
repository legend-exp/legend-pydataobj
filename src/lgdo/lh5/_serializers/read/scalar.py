from __future__ import annotations

import logging

import h5py
import numpy as np

from ....types import Scalar
from ...exceptions import LH5DecodeError
from . import utils

log = logging.getLogger(__name__)


def _h5_read_scalar(
    h5d,
    fname,
    oname,
    obj_buf=None,
):
    value = np.empty((), h5d.dtype)
    sp = h5py.h5s.create(h5py.h5s.SCALAR)
    h5d.read(sp, sp, value)
    value = value[()]
    attrs = utils.read_attrs(h5d, fname, oname)

    # special handling for bools
    # (c and Julia store as uint8 so cast to bool)
    if attrs["datatype"] == "bool":
        value = np.bool_(value)

    if obj_buf is not None:
        if not isinstance(obj_buf, Scalar):
            msg = "object buffer a Scalar"
            raise LH5DecodeError(msg, fname, oname)

        obj_buf.value = value
        obj_buf.attrs.update(attrs)
        return obj_buf, 1

    return Scalar(value=value, attrs=attrs), 1
