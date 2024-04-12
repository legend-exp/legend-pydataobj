from __future__ import annotations

import logging

from .... import types
from ...exceptions import LH5EncodeError

log = logging.getLogger(__name__)


def _h5_write_scalar(obj, name, lh5_file, group="/", wo_mode="append"):
    assert isinstance(obj, types.Scalar)

    if name in group:
        if wo_mode in ["o", "a"]:
            log.debug(f"overwriting {name} in {group}")
            del group[name]
        else:
            msg = f"tried to overwrite but wo_mode is {wo_mode!r}"
            raise LH5EncodeError(msg, lh5_file, group, name)

    ds = group.create_dataset(name, shape=(), data=obj.value)
    ds.attrs.update(obj.attrs)
