from __future__ import annotations

import h5py
import numpy as np

from ...exceptions import LH5DecodeError


def check_obj_buf_attrs(attrs, new_attrs, fname, oname):
    if set(attrs.keys()) != set(new_attrs.keys()):
        msg = (
            f"existing buffer and new data chunk have different attributes: "
            f"obj_buf.attrs={attrs} != {fname}[{oname}].attrs={new_attrs}"
        )
        raise LH5DecodeError(msg, fname, oname)


def read_attrs(h5o, fname, oname):
    """Read all attributes for an hdf5 dataset or group using low level API
    and return them as a dict. Assume all are strings or scalar types."""
    attrs = {}
    for i_attr in range(h5py.h5a.get_num_attrs(h5o)):
        h5a = h5py.h5a.open(h5o, index=i_attr)
        name = h5a.get_name().decode()
        if h5a.shape != ():
            msg = f"attribute {name} is not a string or scalar"
            raise LH5DecodeError(msg, fname, oname)
        val = np.empty((), h5a.dtype)
        h5a.read(val)
        if h5a.get_type().get_class() == h5py.h5t.STRING:
            attrs[name] = val.item().decode()
        else:
            attrs[name] = val.item()
        h5a.close()
    return attrs
