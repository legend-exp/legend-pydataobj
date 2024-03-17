from __future__ import annotations

import logging

import h5py
import numpy as np

from .... import types
from ...exceptions import LH5EncodeError

log = logging.getLogger(__name__)

DEFAULT_HDF5_SETTINGS: dict[str, ...] = {"shuffle": True, "compression": "gzip"}


def _h5_write_array(
    obj,
    name,
    lh5_file,
    group="/",
    start_row=0,
    n_rows=None,
    wo_mode="append",
    write_start=0,
    **h5py_kwargs,
):
    assert isinstance(obj, types.Array)

    if n_rows is None or n_rows > obj.nda.shape[0] - start_row:
        n_rows = obj.nda.shape[0] - start_row

    nda = obj.nda[start_row : start_row + n_rows]

    # hack to store bools as uint8 for c / Julia compliance
    if nda.dtype.name == "bool":
        nda = nda.astype(np.uint8)

    # need to create dataset from ndarray the first time for speed
    # creating an empty dataset and appending to that is super slow!
    if (wo_mode != "a" and write_start == 0) or name not in group:
        # this is needed in order to have a resizable (in the first
        # axis) data set, i.e. rows can be appended later
        # NOTE: this automatically turns chunking on!
        maxshape = (None,) + nda.shape[1:]
        h5py_kwargs.setdefault("maxshape", maxshape)

        if wo_mode == "o" and name in group:
            log.debug(f"overwriting {name} in {group}")
            del group[name]

        # set default compression options
        for k, v in DEFAULT_HDF5_SETTINGS.items():
            h5py_kwargs.setdefault(k, v)

        # compress using the 'compression' LGDO attribute, if available
        if "compression" in obj.attrs:
            comp_algo = obj.attrs["compression"]
            if isinstance(comp_algo, dict):
                h5py_kwargs |= obj.attrs["compression"]
            else:
                h5py_kwargs["compression"] = obj.attrs["compression"]

        # and even the 'hdf5_settings' one, preferred
        if "hdf5_settings" in obj.attrs:
            h5py_kwargs |= obj.attrs["hdf5_settings"]

        # create HDF5 dataset
        ds = group.create_dataset(name, data=nda, **h5py_kwargs)

        # attach HDF5 dataset attributes, but not "compression"!
        _attrs = obj.getattrs(datatype=True)
        _attrs.pop("compression", None)
        _attrs.pop("hdf5_settings", None)
        ds.attrs.update(_attrs)

        return

    # Now append or overwrite
    ds = group[name]
    if not isinstance(ds, h5py.Dataset):
        msg = (
            f"existing HDF5 object '{name}' in group '{group}'"
            " is not a dataset! Cannot overwrite or append"
        )
        raise LH5EncodeError(msg, lh5_file, group, name)

    old_len = ds.shape[0]
    if wo_mode == "a":
        write_start = old_len

    ds.resize(write_start + nda.shape[0], axis=0)
    ds[write_start:] = nda
